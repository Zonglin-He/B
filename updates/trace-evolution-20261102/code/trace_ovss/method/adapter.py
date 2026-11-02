"""Model-facing utilities used by standalone TRACE."""

import copy

import torch
import torch.nn as nn


class ModelAdapterMixin:
    """Adapt only visual LayerNorm affine parameters."""

    def forward_eval_logits(self, x, interpolate, save_weights=False):
        logits, _, _ = self.model(
            x,
            self.text_x[-1],
            True,
            vision_outputs=self.vision_outputs,
            interpolate=interpolate,
            vision_out_type="adaptive_weighted_mean",
            save_weights=save_weights,
        )
        return logits[0]

    def forward_adapt_logits(self, x):
        logits, _, _ = self.model(
            x,
            self.text_x[:-1],
            True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean",
        )
        return logits

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)
                class_embeddings_avg = (
                    class_embeddings_avg
                    / class_embeddings_avg.norm().clamp_min(1e-8)
                )
                class_embeddings = torch.cat(
                    [class_embeddings, class_embeddings_avg.unsqueeze(0)],
                    dim=0,
                )
            text_features.append(class_embeddings)
        return torch.stack(text_features, dim=1).to(self.device)

    @staticmethod
    def set_ln_grads(model):
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params = []
        names = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters():
                    if param_name in ["weight", "bias"] and param.requires_grad:
                        params.append(param)
                        names.append(f"visual.{module_name}.{param_name}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        # Frozen weights never change during episodic LayerNorm adaptation.
        return (
            {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            copy.deepcopy(optimizer.state_dict()),
        )

    @staticmethod
    def load_model_and_optimizer(
        model,
        optimizer,
        model_state,
        optimizer_state,
    ):
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, initial_value in model_state.items():
                parameters[name].copy_(initial_value)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)
