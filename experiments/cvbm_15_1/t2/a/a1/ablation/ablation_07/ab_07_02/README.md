# Ablation 07-02: Foreground Weak, Background Weak

This ablation keeps the a1 Cross-SKC prototype backbone unchanged and feeds the
weakly augmented mixed image to both branches:

- foreground branch input: `image` / `image_weak`
- background branch input: `image` / `image_weak`

Foreground losses, background losses, and branch prototype labels/confidence are
all aligned with the weak augmentation path.
