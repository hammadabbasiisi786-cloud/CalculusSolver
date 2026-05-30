import torch
import torch.nn as nn

RULE_LABELS = [
    "quotient_rule",
    "product_rule",
    "chain_rule",
    "power_rule",
    "sum_rule",
    "partial_x",
    "partial_y",
    "form_lagrangian",
    "solve_system",
    "evaluate_objective",
    "simplify",
    "undefined",
]


class RuleHead(nn.Module):
    def __init__(self, hidden_dim, num_rules=None):
        super().__init__()
        self.num_rules = num_rules or len(RULE_LABELS)
        self.classifier = nn.Linear(hidden_dim, self.num_rules)
        self.rule_embeddings = nn.Embedding(self.num_rules, hidden_dim)

    def forward(self, encoder_out, root_mask=None):
        if root_mask is not None:
            root_mask = root_mask.float().unsqueeze(-1)
            pooled = (encoder_out * root_mask).sum(dim=1) / (
                root_mask.sum(dim=1) + 1e-6
            )
        else:
            pooled = encoder_out[:, 0, :]

        logits = self.classifier(pooled)
        return logits

    def embed_rules(self, rule_ids):
        return self.rule_embeddings(rule_ids)

    @classmethod
    def labels(cls):
        return list(RULE_LABELS)
