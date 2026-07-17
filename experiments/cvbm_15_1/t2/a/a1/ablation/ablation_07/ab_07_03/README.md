# Ablation 07-03: Foreground Strong, Background Strong

This ablation keeps the a1 Cross-SKC prototype backbone unchanged and feeds the
strongly augmented mixed image to both branches:

- foreground branch input: `image_strong`
- background branch input: `image_strong`

Foreground losses, background losses, and branch prototype labels/confidence are
all aligned with the strong augmentation path.
