import torch
import torch.nn as nn

RULE_DESCRIPTION_TEMPLATES = {
    "quotient_rule": "Differentiate the numerator and denominator separately using the quotient rule.",
    "product_rule": "Apply the product rule to differentiate both factors.",
    "chain_rule": "Use the chain rule for the nested function composition.",
    "power_rule": "Differentiate the power expression using the power rule.",
    "sum_rule": "Differentiate each term in the sum independently.",
    "partial_x": "Compute the partial derivative with respect to x.",
    "partial_y": "Compute the partial derivative with respect to y.",
    "form_lagrangian": "Build the Lagrangian for the constrained optimization problem.",
    "solve_system": "Solve the resulting system of equations from the Lagrange conditions.",
    "evaluate_objective": "Evaluate the objective function at the candidate critical points.",
    "simplify": "Simplify the expression algebraically.",
    "undefined": "Mark the expression as undefined or non-differentiable.",
}


class StepTracer(nn.Module):
    def __init__(self, hidden_dim=512, num_templates=None):
        super().__init__()
        self.templates = list(RULE_DESCRIPTION_TEMPLATES.values())
        self.num_templates = num_templates or len(self.templates)
        self.classifier = nn.Linear(hidden_dim, self.num_templates)

    def forward(self, rule_ids, decoder_hidden_states):
        if decoder_hidden_states is not None:
            hidden = decoder_hidden_states.mean(dim=1)
        else:
            hidden = torch.nn.functional.one_hot(
                rule_ids, num_classes=self.num_templates
            ).float()

        logits = self.classifier(hidden)
        return logits

    def template_for(self, rule_id):
        template_index = (
            rule_id.item() if isinstance(rule_id, torch.Tensor) else rule_id
        )
        return self.templates[template_index]
