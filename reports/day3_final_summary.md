# Day 3 final evaluation

Frozen checkpoint: `checkpoints\day3_deployed.pt`. Official MELD test examples: 2610.

| Model | Macro-F1 | Weighted F1 | Accuracy |
|---|---:|---:|---:|
| Deployed fusion | 0.4032 | 0.5707 | 0.5720 |
| Text only | 0.4023 | 0.5648 | 0.5632 |
| Audio only | 0.2228 | 0.3856 | 0.3805 |
| Majority (neutral) | 0.0928 | 0.3127 | 0.4812 |

Audio shuffle macro-F1: 0.4045. Dialogue-bootstrap fused-minus-text 95% CI: [-0.0088, 0.0096].
