import base64
import json
import zlib
import numpy as np
from scipy.stats import wasserstein_distance
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from utils.fipca import ServerFIPCA, reconstruct_delta, flatten_weights, unflatten_weights

class DistanceAwareAggregation(FedAvg):
    def __init__(
        self,
        *args,
        similarity_metric="kl",
        target_components=50,
        dp_epsilon=0.0,
        dp_clip_norm=1.0,
        compression_warmup_rounds=6,
        min_pca_components=8,
        max_recon_error=0.05,
        distance_strength=0.35,
        min_weight=0.20,
        max_weight=0.45,
        server_lr_compressed=0.50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_reference_dist = None
        self.similarity_metric = similarity_metric
        self.fipca = ServerFIPCA(target_components=target_components)
        self.global_weights_flat = None
        self.param_shapes = None
        self.current_basis_version = 0
        self.basis_just_changed = False
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm
        self.compression_warmup_rounds = compression_warmup_rounds
        self.min_pca_components = min_pca_components
        self.max_recon_error = max_recon_error
        self.distance_strength = distance_strength
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.server_lr_compressed = server_lr_compressed

    def configure_fit(self, server_round, parameters, client_manager):
        base = super().configure_fit(server_round, parameters, client_manager)
        dp_config = {"dp_epsilon": str(self.dp_epsilon), "dp_clip_norm": str(self.dp_clip_norm)}
        compression_ready = (
            self.fipca.is_fitted()
            and server_round > self.compression_warmup_rounds
            and self.fipca.n_components_current >= self.min_pca_components
        )
        if not compression_ready:
            config = {
                "pca_fitted": "0",
                "server_round": str(server_round),
                "pca_min_components": str(self.min_pca_components),
                "pca_max_recon_error": str(self.max_recon_error),
                **dp_config,
            }
            return [(cp, FitIns(fi.parameters, config)) for cp, fi in base]

        config = {
            "pca_fitted": "1",
            "basis_version": str(self.current_basis_version),
            "server_round": str(server_round),
            "pca_min_components": str(self.min_pca_components),
            "pca_max_recon_error": str(self.max_recon_error),
            **dp_config,
        }
        if self.basis_just_changed:
            b64, n_k_str, d_str = self.fipca.get_basis_b64()
            config["pca_components"] = b64
            config["n_components"]   = n_k_str
            config["original_dim"]   = d_str
        return [(cp, FitIns(fi.parameters, config)) for cp, fi in base]

    def _compute_distances(self, dist_array):
        eps = 1e-12
        dist_array = np.asarray(dist_array, dtype=np.float64)
        reference = np.asarray(self.server_reference_dist, dtype=np.float64)

        dist_array = np.clip(dist_array, eps, None)
        dist_array = dist_array / np.sum(dist_array, axis=1, keepdims=True)

        reference = np.clip(reference, eps, None)
        reference = reference / np.sum(reference)

        if self.similarity_metric == "wasserstein":
            class_idx = np.arange(dist_array.shape[1], dtype=np.float64)
            distances = np.array([
                wasserstein_distance(
                    class_idx,
                    class_idx,
                    u_weights=dist_array[i],
                    v_weights=reference,
                )
                for i in range(len(dist_array))
            ], dtype=np.float64)
        elif self.similarity_metric == "js":
            midpoint = 0.5 * (dist_array + reference)
            distances = (
                0.5 * np.sum(dist_array * np.log(dist_array / midpoint), axis=1)
                + 0.5 * np.sum(reference * np.log(reference / midpoint), axis=1)
            )
        else:
            distances = np.sum(dist_array * np.log(dist_array / reference), axis=1)

        distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=0.0)
        return np.maximum(distances, 0.0)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        try:
            weights_list = []
            dist_list = []
            compressed_flags = []
            recon_errors = []
            missing_label_dist = False

            for client, fit_res in results:
                compressed = fit_res.metrics.get("compressed", "0") == "1"
                compressed_flags.append(compressed)
                recon_error = float(fit_res.metrics.get("recon_error", "nan"))
                if np.isfinite(recon_error):
                    recon_errors.append(recon_error)
                if compressed and self.fipca.is_fitted() and self.global_weights_flat is not None:
                    scores = parameters_to_ndarrays(fit_res.parameters)[0]
                    full_flat = reconstruct_delta(scores, self.fipca.pca.components_, self.fipca.pca.mean_)
                    weights = unflatten_weights(full_flat, self.param_shapes)
                else:
                    weights = parameters_to_ndarrays(fit_res.parameters)
                    if self.param_shapes is None and weights:
                        self.param_shapes = [a.shape for a in weights]
                    if self.global_weights_flat is None and weights:
                        self.global_weights_flat = flatten_weights(weights)
                weights_list.append(weights)
                if fit_res.metrics and "label_dist" in fit_res.metrics:
                    dist_list.append(json.loads(fit_res.metrics["label_dist"]))
                else:
                    missing_label_dist = True

            n_clients = len(weights_list)
            if n_clients == 0:
                return None, {}

            if missing_label_dist or len(dist_list) != n_clients:
                agg_weights = np.ones(n_clients, dtype=np.float64) / n_clients
                distances = np.zeros(n_clients, dtype=np.float64)
                weight_status = "uniform fallback: missing label_dist"
            else:
                dist_array = np.asarray(dist_list, dtype=np.float64)
                self.server_reference_dist = np.mean(dist_array, axis=0)

                distances = self._compute_distances(dist_array)
                distance_span = np.max(distances) - np.min(distances)

                if distance_span < 1e-12:
                    raw_weights = np.ones(n_clients, dtype=np.float64) / n_clients
                else:
                    scaled_distances = (distances - np.min(distances)) / (distance_span + 1e-12)
                    temperature = 1.0
                    logits = -scaled_distances / temperature
                    logits = logits - np.max(logits)
                    exp_scores = np.exp(logits)
                    raw_weights = exp_scores / np.sum(exp_scores)

                uniform_weights = np.ones(n_clients, dtype=np.float64) / n_clients
                raw_weights = (1.0 - self.distance_strength) * uniform_weights + self.distance_strength * raw_weights

                min_weight = self.min_weight
                max_weight = self.max_weight

                if n_clients * min_weight > 1.0:
                    min_weight = 1.0 / n_clients
                if n_clients * max_weight < 1.0:
                    max_weight = 1.0 / n_clients

                agg_weights = np.clip(raw_weights, min_weight, max_weight)

                for _ in range(20):
                    diff = 1.0 - np.sum(agg_weights)
                    if abs(diff) < 1e-12:
                        break

                    if diff > 0:
                        room = max_weight - agg_weights
                        eligible = room > 1e-12
                        if not np.any(eligible):
                            break
                        agg_weights[eligible] += diff * room[eligible] / np.sum(room[eligible])
                    else:
                        room = agg_weights - min_weight
                        eligible = room > 1e-12
                        if not np.any(eligible):
                            break
                        agg_weights[eligible] += diff * room[eligible] / np.sum(room[eligible])

                    agg_weights = np.clip(agg_weights, min_weight, max_weight)

                agg_weights = agg_weights / np.sum(agg_weights)
                weight_status = "bounded distance-aware"

            candidate_global_weights = [np.average(np.stack(layer), axis=0, weights=agg_weights) for layer in zip(*weights_list)]
            candidate_global_flat = flatten_weights(candidate_global_weights)
            compressed_clients = int(sum(compressed_flags))

            if compressed_clients > 0 and self.global_weights_flat is not None:
                server_lr = self.server_lr_compressed
                new_global_flat = self.global_weights_flat + server_lr * (candidate_global_flat - self.global_weights_flat)
                global_weights = unflatten_weights(new_global_flat, self.param_shapes)
            else:
                server_lr = 1.0
                new_global_flat = candidate_global_flat
                global_weights = candidate_global_weights

            print(f"[DEBUG] Round {server_round} | new_global_flat.shape = {new_global_flat.shape}, dtype = {new_global_flat.dtype}")
            print(f"[DEBUG] Round {server_round} | history_len before update = {len(self.fipca.history)}")
            prev_k = self.fipca.n_components_current
            pca_input = new_global_flat
            self.fipca.update(pca_input)
            self.global_weights_flat = new_global_flat
            if self.fipca.is_fitted() and self.fipca.n_components_current != prev_k:
                self.current_basis_version += 1
                self.basis_just_changed = True
            else:
                self.basis_just_changed = self.fipca.basis_changed()
                if self.basis_just_changed:
                    self.current_basis_version += 1

            bytes_before_list = [int(r.metrics.get("bytes_before", "0")) for _, r in results]
            bytes_after_list  = [int(r.metrics.get("bytes_after",  "0")) for _, r in results]
            avg_before = sum(bytes_before_list) / len(bytes_before_list) if bytes_before_list else 0
            avg_after  = sum(bytes_after_list)  / len(bytes_after_list)  if bytes_after_list  else 0
            if avg_before > 0:
                reduction = (1.0 - avg_after / avg_before) * 100.0
                k_str = str(self.fipca.n_components_current) if self.fipca.is_fitted() else "—"
                status = f"K={k_str} | {avg_after/1024:.3f} KB sent vs {avg_before/1024:.1f} KB | Reduction: {reduction:.1f}%"
            else:
                status = "fallback (PCA warming up)"
            avg_recon_error = float(np.mean(recon_errors)) if recon_errors else float("nan")
            max_recon_error = float(np.max(recon_errors)) if recon_errors else float("nan")
            compression_rate = compressed_clients / n_clients if n_clients else 0.0
            print(
                f"[DAA] Round {server_round} | mode={weight_status} | "
                f"weights={agg_weights.tolist()} | compressed={compressed_clients}/{n_clients} | "
                f"avg_recon_error={avg_recon_error:.6f} | max_recon_error={max_recon_error:.6f} | "
                f"server_lr={server_lr}"
            )
            print(f"[FIPCA] Round {server_round} | {status}")
            print(f"[DEBUG] Round {server_round} aggregate_fit completed successfully.")

            return ndarrays_to_parameters(global_weights), {
                "server_round":   server_round,
                "client_weights": agg_weights.tolist(),
                "compressed_clients": compressed_clients,
                "compression_rate": compression_rate,
                "avg_recon_error": avg_recon_error,
                "max_recon_error": max_recon_error,
                "server_lr": server_lr,
                "metric":         self.similarity_metric,
            }

        except Exception:
            import traceback
            print(f"\n\n!!! AGGREGATE_FIT CRASHED IN ROUND {server_round} !!!")
            traceback.print_exc()
            raise
