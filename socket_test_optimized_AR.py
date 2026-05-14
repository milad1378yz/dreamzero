import dataclasses
import json
import logging
import socket
import asyncio
import os
import http
import logging
import sys
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

# Make the parent repo importable so we can use wam_search.* without
# polluting the dreamzero environment with extra deps. wam_search only
# depends on numpy, which is already required by this file.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wam_search.executable_reward import executable_action_score  # noqa: E402

try:
    from prm.reward import PRMReward, PRMRewardConfig  # noqa: E402
except Exception:  # pragma: no cover - prm package optional at import time
    PRMReward = None  # type: ignore[assignment]
    PRMRewardConfig = None  # type: ignore[assignment]

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

def _apply_max_chunk_size(policy: GrootSimPolicy, max_chunk_size: int | None) -> None:
    if max_chunk_size is None:
        return

    dit_model = policy.trained_model.action_head.model
    local_attn_size = (
        max_chunk_size * dit_model.num_frame_per_block + 1
        if max_chunk_size != -1
        else -1
    )
    dit_model.local_attn_size = local_attn_size

    for block in dit_model.blocks:
        block.local_attn_size = local_attn_size
        block.self_attn.local_attn_size = local_attn_size
        block.self_attn.max_attention_size = (
            21 * dit_model.frame_seqlen
            if local_attn_size == -1
            else local_attn_size * dit_model.frame_seqlen
        )

    logger.info(
        "Set DiT max_chunk_size=%s, local_attn_size=%s",
        max_chunk_size,
        local_attn_size,
    )

@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/dreamzero"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.
    attention_backend: str = "FA2"
    quantization: str | None = None
    # Best-of-K reranking (EVA-style executability score).
    # K=1 reproduces the original single-forward behavior.
    search_k: int = 1
    search_log_path: str | None = None
    # When True (default), set torch.manual_seed per candidate so the
    # diffusion sampler produces different rollouts across the K passes.
    # Seeds are identical across ranks to keep the distributed forward
    # consistent.
    search_seed_per_candidate: bool = True

    # PRM (learned IDM-based) reward. If prm_idm_ckpt is set, the K-candidate
    # reranker decodes each candidate's predicted video, runs the IDM, and
    # scores with R_exec + lambda_cons * R_cons in place of the heuristic
    # executable_action_score. Falls back to the heuristic on any error.
    prm_idm_ckpt: str | None = None
    prm_lambda_exec: float = 1.0
    prm_lambda_cons: float = 1.0

    # Embodiment selection. "oxe_droid" preserves the original DROID-only
    # behavior. "libero" switches camera key remapping, frame buffers, and
    # PolicyServerConfig to the 2-cam (agentview + wrist) LIBERO layout.
    embodiment_tag: str = "oxe_droid"


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.
    
    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """
    
    # Number of frames to accumulate after the first call
    FRAMES_PER_CHUNK = 4
    
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        search_k: int = 1,
        search_log_path: str | None = None,
        search_seed_per_candidate: bool = True,
        prm_idm_ckpt: str | None = None,
        prm_lambda_exec: float = 1.0,
        prm_lambda_cons: float = 1.0,
        embodiment_tag: str = "oxe_droid",
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir

        self._search_k = max(int(search_k), 1)
        self._search_log_path = search_log_path
        self._search_seed_per_candidate = bool(search_seed_per_candidate)
        self._embodiment_tag = embodiment_tag

        self._prm_reward = None
        if prm_idm_ckpt:
            if PRMReward is None or PRMRewardConfig is None:
                logger.warning(
                    "prm package not importable; falling back to heuristic reward."
                )
            else:
                try:
                    self._prm_reward = PRMReward(
                        PRMRewardConfig(
                            idm_ckpt_path=prm_idm_ckpt,
                            lambda_exec=float(prm_lambda_exec),
                            lambda_cons=float(prm_lambda_cons),
                        )
                    )
                    logger.info(
                        "PRM reward enabled: ckpt=%s exec=%.3f cons=%.3f",
                        prm_idm_ckpt, prm_lambda_exec, prm_lambda_cons,
                    )
                except Exception as exc:  # pragma: no cover - runtime fallback
                    logger.warning(
                        "Failed to load PRM reward (%s); falling back to heuristic.", exc
                    )
                    self._prm_reward = None

        # Embodiment-specific camera key remapping. The websocket protocol
        # sends roboarena-style keys ("observation/exterior_image_*", etc.);
        # the DreamZero data pipeline expects modality-prefixed keys whose
        # exact names differ per embodiment (DROID = 3 cams, LIBERO = 2 cams).
        if self._embodiment_tag == "libero":
            self._image_key_mapping: dict[str, str] = {
                "observation/exterior_image_0_left": "video.image",
                "observation/wrist_image_left": "video.wrist_image",
            }
        else:
            self._image_key_mapping = {
                "observation/exterior_image_0_left": "video.exterior_image_1_left",
                "observation/exterior_image_1_left": "video.exterior_image_2_left",
                "observation/wrist_image_left": "video.wrist_image_left",
            }
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            droid_key: [] for droid_key in self._image_key_mapping.values()
        }
        self._call_count = 0
        self._is_first_call = True

        # Session tracking - reset state when new session starts
        self._current_session_id: str | None = None

        # Video across time for saving (similar to original server)
        self.video_across_time = []
        # Mirror list for the *worst*-scoring candidate at each infer() call
        # when K > 1; used to render a side-by-side comparison video on reset.
        self._worst_video_across_time = []
        self._msg_index = 0

        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
        if self._search_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._search_log_path)), exist_ok=True)

    def _decode_video_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        action_head = self._policy.trained_model.action_head
        if hasattr(action_head, "_ensure_vae_on_device"):
            action_head._ensure_vae_on_device(video_latents)
        else:
            action_head.vae.to(device=video_latents.device, dtype=video_latents.dtype)

        try:
            return action_head.vae.decode(
                video_latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
        finally:
            if hasattr(action_head, "_offload_auxiliary_components"):
                action_head._offload_auxiliary_components()

    def _score_candidate(self, action: np.ndarray, video_pred: torch.Tensor) -> tuple[float, dict]:
        """Score one best-of-K candidate.

        If a PRM (trained IDM) is loaded, run R_exec + lambda_cons * R_cons on
        the decoded predicted video. Otherwise use the heuristic
        executable_action_score on the action chunk only.
        """
        if self._prm_reward is None:
            return executable_action_score(action)

        try:
            frames = self._video_pred_to_idm_frames(video_pred)
            return self._prm_reward.score(action, frames)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning(
                "PRM scoring failed (%s); falling back to heuristic for this candidate.", exc
            )
            return executable_action_score(action)

    def _video_pred_to_idm_frames(self, video_pred: torch.Tensor) -> np.ndarray:
        """Convert DreamZero predicted video latents to IDM input frames.

        DreamZero's data pipeline tile-stitches multiple cameras into a single
        2×2 canvas before encoding (see ``groot.vla.model.dreamzero.transform.
        dreamzero_cotrain._prepare_video``). So the decoded video has the
        cameras laid out spatially:

            ┌──────────────────┬──────────────────┐
            │ view 0  (TL)     │ view 2  (TR)     │
            ├──────────────────┼──────────────────┤
            │ view 1  (BL)     │ black   (BR)     │
            └──────────────────┴──────────────────┘

        For LIBERO-90 the LoRA was trained on 2 cams: view 0 = ``video.image``
        (agentview), view 1 = ``video.wrist_image``. So we extract TL + BL.

        Returns ``(n_cams, T, 3, H, W)`` uint8 resized to the IDM's
        ``image_size``. The aspect-ratio rescale from the (h, 2w → 1:2) tile to
        the IDM's square input is a known minor distortion; the IDM was trained
        on LIBERO's native 128×128 frames and should generalize, but if the
        consistency reward looks degenerate at inference, retrain the IDM at
        the LoRA's tile aspect ratio.
        """
        from prm.reward import prepare_video_frames_for_idm

        decoded = self._decode_video_latents(video_pred)
        decoded = rearrange(decoded, "B C T H W -> B T H W C")
        decoded = ((decoded.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

        if decoded.shape[0] != 1:
            raise ValueError(
                f"expected B=1 in decoded video (tile-stitched), got {decoded.shape[0]}"
            )
        canvas = decoded[0]  # (T, 2h, 2w, 3)
        t_full, h2, w2, _ = canvas.shape
        h = h2 // 2
        w = w2 // 2

        n_cams = self._prm_reward.cfg.n_cams
        if n_cams == 1:
            cams = canvas[None, :, :h, :w, :]  # agentview only
        elif n_cams == 2:
            agent = canvas[:, :h, :w, :]       # top-left
            wrist = canvas[:, h:, :w, :]       # bottom-left
            cams = np.stack([agent, wrist], axis=0)  # (2, T, h, w, 3)
        elif n_cams == 3:
            agent = canvas[:, :h, :w, :]
            wrist = canvas[:, h:, :w, :]
            third = canvas[:, :h, w:, :]
            cams = np.stack([agent, wrist, third], axis=0)
        else:
            raise ValueError(f"n_cams={n_cams} not supported by 2x2 tile extractor")

        return prepare_video_frames_for_idm(
            cams,
            n_cams=n_cams,
            video_len=self._prm_reward.cfg.video_len,
            image_size=self._prm_reward.cfg.target_image_size,
        )
    
    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.
        
        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str
        
        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}
        
        # Embodiment-specific image key remapping was built in __init__.
        image_key_mapping = self._image_key_mapping

        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK
        
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
        
        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
        
        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""
        
        return converted
    
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action
    
    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle
        
        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)
        
        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        
        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)
    
    def _snapshot_action_head_state(self) -> dict | None:
        """Clone the action head's mutable AR state so K candidates can fork
        from the same point. Returns None if the head is missing the expected
        attributes (best-effort; we fall back to the original drift behavior).

        Snapshots `current_start_frame` and the two KV-cache lists. Tensors
        are moved to CPU so the snapshot doesn't double GPU KV-cache memory
        during best-of-K — we pay a small PCIe cost per candidate restore in
        exchange for fitting on tight VRAM (e.g. 24 GB 4090s).
        """
        try:
            head = self._policy.trained_model.action_head
        except AttributeError:
            return None
        snap = {"current_start_frame": int(getattr(head, "current_start_frame", 0))}
        for name in ("kv_cache1", "kv_cache_neg"):
            cache = getattr(head, name, None)
            if cache is None:
                snap[name] = None
            else:
                # .to("cpu") on a CUDA tensor returns a CPU copy; pin to speed
                # up subsequent .cuda() restores.
                snap[name] = [t.detach().to("cpu", copy=True).pin_memory() for t in cache]
        return snap

    def _restore_action_head_state(self, snap: dict | None) -> None:
        if snap is None:
            return
        try:
            head = self._policy.trained_model.action_head
        except AttributeError:
            return
        head.current_start_frame = snap["current_start_frame"]
        for name in ("kv_cache1", "kv_cache_neg"):
            cached = snap.get(name)
            if cached is None:
                setattr(head, name, None)
            else:
                # Move CPU-resident snapshots back to GPU for the next forward.
                setattr(
                    head,
                    name,
                    [t.to("cuda", non_blocking=True) for t in cached],
                )

    def _extract_action_array(self, result_batch) -> np.ndarray:
        """Pull the (N, 8) action array out of the policy result batch."""
        action_chunk_dict = result_batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        return self._convert_action(action_dict)

    def _distributed_forward_once(self, converted_obs: dict, seed: int | None = None):
        """Run one distributed forward over an already-converted observation.

        All ranks must call into the same forward, so this re-broadcasts the
        obs (and optional seed) every iteration. The signal broadcast is done
        once per infer() call by the caller, not here.

        DreamZero's action head hardcodes self.seed=1140, which is passed into
        `torch.Generator().manual_seed(...)` for diffusion noise. To get
        different rollouts per candidate we override that attribute directly
        (mirroring the same override in the worker loop so all ranks sample
        identical noise).
        """
        # Re-broadcast obs for this candidate.
        self._broadcast_batch_to_workers(converted_obs)

        if seed is not None:
            action_head = self._policy.trained_model.action_head
            action_head.seed = int(seed)
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        batch = Batch(obs=converted_obs)

        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        return result_batch, video_pred

    def _log_search_candidates(self, candidates: list[dict], best_idx: int) -> None:
        if not self._search_log_path:
            return

        row = {
            "time": time.time(),
            "call_count": self._call_count,
            "msg_index": self._msg_index,
            "best_idx": int(best_idx),
            "num_candidates": len(candidates),
            "candidates": [
                {
                    "idx": int(c["idx"]),
                    "score": float(c["score"]),
                    "seed": c.get("seed"),
                    "score_info": c["score_info"],
                    "action_mean": float(np.mean(c["action"])),
                    "action_std": float(np.std(c["action"])),
                    "action_first": c["action"][0].tolist(),
                    "action_last": c["action"][-1].tolist(),
                }
                for c in candidates
            ],
        }

        try:
            with open(self._search_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write search log: {e}")

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations with optional best-of-K reranking.

        K is taken from obs["search/k"] if present, otherwise self._search_k.

        Args:
            obs: Observation dict in roboarena format

        Returns:
            action: (N, 8) action array — the highest-scoring candidate.
        """
        # Check for session change - reset state if new session
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        # Convert observation once; we will reuse it across the K candidates.
        converted_obs = self._convert_observation(obs)

        # Resolve K (per-request override beats the server-wide default).
        k_override = obs.get("search/k", None)
        k = self._search_k if k_override is None else max(int(k_override), 1)

        # Signal workers to continue (0 = continue) — once per infer() call.
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast (K, base_seed) so workers know how many forwards to run
        # and which seed to use. base_seed varies per call to keep diversity
        # between successive infer() invocations as well.
        base_seed = (int(time.time() * 1000) ^ (self._call_count * 1315423911)) & 0x7FFFFFFF
        kseed_tensor = torch.tensor([int(k), int(base_seed)], dtype=torch.int64, device='cpu')
        dist.broadcast(kseed_tensor, src=0, group=self._signal_group)

        # Snapshot AR state so every candidate forks from the same point
        # rather than chaining off the previous one's KV-cache mutations.
        # Snapshot only when K > 1; the single-candidate path is unchanged.
        snap = self._snapshot_action_head_state() if k > 1 else None

        candidates: list[dict] = []
        for i in range(k):
            if k > 1 and i > 0:
                self._restore_action_head_state(snap)
            seed = (base_seed + i) if (self._search_seed_per_candidate and k > 1) else None
            result_batch, video_pred = self._distributed_forward_once(converted_obs, seed=seed)
            action = self._extract_action_array(result_batch)
            score, score_info = self._score_candidate(action, video_pred)
            candidates.append({
                "idx": i,
                "seed": seed,
                "score": score,
                "score_info": score_info,
                "action": action,
                "video_pred": video_pred,
            })

        # After picking the winner, drop the snapshot tensors so they don't
        # linger as a memory ceiling across infer calls.
        snap = None

        best = max(candidates, key=lambda c: c["score"])

        # Only keep the winning rollout's video for downstream saving.
        self.video_across_time.append(best["video_pred"])

        # When K > 1, also keep the *worst* rollout so we can render a
        # best-vs-worst comparison video at reset. With K = 1 there is
        # nothing to compare against, so leave the worst list empty.
        if len(candidates) > 1:
            worst = min(candidates, key=lambda c: c["score"])
            self._worst_video_across_time.append(worst["video_pred"])

        self._log_search_candidates(candidates, best["idx"])

        if self._is_first_call:
            self._is_first_call = False

        return best["action"]
    
    def _save_video_list(self, video_list: list, tag: str) -> str | None:
        """Decode a list of video-pred latents and write a single MP4.

        Returns the output path, or None on failure / no-op.
        """
        if not video_list or not self._output_dir:
            return None
        try:
            video_cat = torch.cat(video_list, dim=2)
            frames = self._decode_video_latents(video_cat)
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = frames[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            frame_list = [f for f in frames]
            if not frame_list:
                return None
            sample = frame_list[0]
            if not (len(sample.shape) == 3 and sample.shape[2] in [1, 3, 4]):
                return None
            os.makedirs(self._output_dir, exist_ok=True)
            existing = [f for f in os.listdir(self._output_dir) if f.endswith(".mp4")]
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            n = (len(frame_list) - 1) // 8
            output_path = os.path.join(
                self._output_dir,
                f"{len(existing):06}_{timestamp}_n{n}_{tag}.mp4",
            )
            imageio.mimsave(output_path, frame_list, fps=5, codec="libx264")
            logger.info(f"Saved {tag} video to: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Failed to save {tag} video: {e}")
            return None

    def _reset_state(self, save_video: bool = True) -> None:
        """Internal method to reset policy state.

        Args:
            save_video: Whether to save accumulated video before reset.
        """
        if save_video:
            self._save_video_list(self.video_across_time, "best")
            self._save_video_list(self._worst_video_across_time, "worst")

        # Clear frame buffers
        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []
        self._worst_video_across_time = []
    
    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.
        
        Clears frame buffers and resets call count.
        """
        self._reset_state(save_video=True)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)
    
    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.
        
        Expected format: THWC (Time, Height, Width, Channel) with 4 frames.
        Saves each frame as a separate PNG image: HWC format (uint8).
        
        Directory structure:
        output_dir/inputs/{msg_index:06d}_{timestamp}/{obs_key}/f{frame_idx:02d}.png
        """
        if not self._output_dir:
            return
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        base_dir = os.path.join(self._output_dir, "inputs", f"{self._msg_index:06d}_{timestamp}")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            return

        for key in ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"):
            if key not in obs:
                continue
            value = obs[key]
            try:
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)
                
                # Expected format: THWC (Time, Height, Width, Channel)
                if arr.ndim != 4:
                    logger.warning(f"obs key '{key}' has shape {arr.shape}, expected 4D (T,H,W,C)")
                    continue
                
                # arr is (T, H, W, C)
                T, H, W, C = arr.shape
                
                # Normalize to uint8
                if arr.dtype == np.uint8:
                    frames_u8 = arr
                else:
                    f = arr.astype(np.float32)
                    # Common conventions: [-1,1] or [0,1]
                    min_val = float(np.nanmin(f))
                    max_val = float(np.nanmax(f))
                    if min_val >= -1.1 and max_val <= 1.1:
                        # Assume [-1,1] range
                        frames_u8 = ((f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        # Min-max scaling
                        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
                        frames_u8 = ((f - min_val) / denom * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save each frame: frames_u8[i] is (H, W, C)
                key_dir = os.path.join(base_dir, key.replace("/", "_"))
                os.makedirs(key_dir, exist_ok=True)
                for frame_idx in range(T):
                    frame = frames_u8[frame_idx]  # (H, W, C)
                    # Handle grayscale (H, W) -> (H, W, 1)
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=-1)
                    imageio.imwrite(os.path.join(key_dir, f"f{frame_idx:02d}.png"), frame)
                    
            except Exception as e:
                logger.warning(f"Failed to save obs key '{key}': {e}")
                continue



    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            # Non-rank-0 processes run a worker loop
            await self._worker_loop()

    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference.

        Per signal from rank 0, this worker receives a (K, base_seed) header and
        then participates in K forward passes — one per best-of-K candidate.
        K=1 reproduces the original single-forward behavior.
        """
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        kseed_tensor = torch.zeros(2, dtype=torch.int64, device='cpu')
        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    continue

                # Receive (K, base_seed) for this infer() call.
                dist.broadcast(kseed_tensor, src=0, group=self._signal_group)
                k = int(kseed_tensor[0].item())
                base_seed = int(kseed_tensor[1].item())
                if k < 1:
                    k = 1

                # Snapshot the action head's AR state on this rank so every
                # candidate forks from the same KV cache. Same logic that the
                # rank-0 wrapper runs in ARDroidRoboarenaPolicy. Snapshots
                # live on CPU to save GPU memory during best-of-K.
                worker_snap = None
                if k > 1:
                    try:
                        head = self._policy.trained_model.action_head
                        worker_snap = {
                            "current_start_frame": int(getattr(head, "current_start_frame", 0)),
                            "kv_cache1": (
                                [t.detach().to("cpu", copy=True).pin_memory() for t in head.kv_cache1]
                                if getattr(head, "kv_cache1", None) is not None else None
                            ),
                            "kv_cache_neg": (
                                [t.detach().to("cpu", copy=True).pin_memory() for t in head.kv_cache_neg]
                                if getattr(head, "kv_cache_neg", None) is not None else None
                            ),
                        }
                    except AttributeError:
                        worker_snap = None

                for i in range(k):
                    if k > 1 and i > 0 and worker_snap is not None:
                        try:
                            head = self._policy.trained_model.action_head
                            head.current_start_frame = worker_snap["current_start_frame"]
                            if worker_snap["kv_cache1"] is None:
                                head.kv_cache1 = None
                            else:
                                head.kv_cache1 = [t.to("cuda", non_blocking=True) for t in worker_snap["kv_cache1"]]
                            if worker_snap["kv_cache_neg"] is None:
                                head.kv_cache_neg = None
                            else:
                                head.kv_cache_neg = [t.to("cuda", non_blocking=True) for t in worker_snap["kv_cache_neg"]]
                        except AttributeError:
                            pass

                    batch = self._receive_batch_from_rank0()
                    if k > 1:
                        seed = base_seed + i
                        try:
                            self._policy.trained_model.action_head.seed = int(seed)
                        except AttributeError:
                            pass
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    dist.barrier()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()

                worker_snap = None

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        """Receive batch data from rank 0 using torch.distributed primitives."""
        import pickle

        # Receive the size of the pickled data first
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        # Receive the actual data
        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        # Deserialize
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs):
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle

        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    # Signal other ranks to continue (0 = continue)
                    signal_tensor.zero_() 
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) # <-- USE GLOO GROUP

                    # Broadcast the obs to all ranks for distributed inference
                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    # All ranks need to participate in the forward pass
                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    self.video_across_time.append(video_chunk)

                    if len(self.video_across_time) > 10:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._decode_video_latents(video_across_time_cat)
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        self.video_across_time = []
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time[:-1], dim=2)
                        frames = self._decode_video_latents(video_across_time_cat)
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        self.video_across_time = [video_chunk]

                    
                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    if len(self.video_across_time) > 0:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._decode_video_latents(video_across_time_cat)
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)  # Set tensor value to 2
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            # When connection closes, signal other ranks to continue waiting for next connection
            # (or implement proper shutdown if needed)


def init_mesh() -> DeviceMesh:
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def main(args: Args) -> None:
    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    os.environ["ATTENTION_BACKEND"] = args.attention_backend
    if args.quantization:
        os.environ.setdefault("DREAMZERO_OFFLOAD_AUXILIARY_COMPONENTS", "true")

    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = args.embodiment_tag
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        quantization=args.quantization,
    )
    _apply_max_chunk_size(policy, args.max_chunk_size)

    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        # Create output directory for videos
        # Extract parent directory and checkpoint name from model_path
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")
    
    # Create wrapper policy that converts between roboarena and AR_droid formats
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
        search_k=args.search_k,
        search_log_path=args.search_log_path,
        search_seed_per_candidate=args.search_seed_per_candidate,
        prm_idm_ckpt=args.prm_idm_ckpt,
        prm_lambda_exec=args.prm_lambda_exec,
        prm_lambda_cons=args.prm_lambda_cons,
        embodiment_tag=embodiment_tag,
    )

    if embodiment_tag == "libero":
        # LIBERO: agentview + wrist (2 cams), 320x176 frames per the LoRA conf.
        server_config = PolicyServerConfig(
            image_resolution=(176, 320),
            needs_wrist_camera=True,
            n_external_cameras=1,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )
    else:
        # AR_droid: 2 external cameras + wrist (3 cams), 180x320 frames.
        server_config = PolicyServerConfig(
            image_resolution=(180, 320),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )
    
    if rank == 0:
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 processes need to run worker loop for distributed inference
        # We'll use the existing WebsocketPolicyServer's worker loop mechanism
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
