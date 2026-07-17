# Ablation 07-01: Foreground Strong, Background Weak

This ablation keeps the a1 Cross-SKC prototype backbone unchanged and swaps the
augmentation binding between the two branches:

- foreground branch input: `image_strong`
- background branch input: `image` / `image_weak`

The foreground losses and prototype labels are aligned with the strong branch,
while the background losses and prototype labels are aligned with the weak
branch.
