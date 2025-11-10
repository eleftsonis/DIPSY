# DIPSY: Dual IP-Adapter Synthesizer

Official repository for the BMVC 2025 paper:  
**"Training-Free Synthetic Data Generation with Dual IP-Adapter Guidance"**

**[➡️ Read the Paper on arXiv](https://arxiv.org/abs/2509.22635)** | **[🌐 View the Project Website](https://www.lix.polytechnique.fr/vista/projects/2025_bmvc_dipsy/)**

---

**Authors:** Luc Boudier, Loris Manganelli, Eleftherios Tsonis, Nicolas Dufour, Vicky Kalogeiton


---

### Abstract
> Few-shot image classification remains challenging due to the limited availability of labeled examples. Recent approaches have explored generating synthetic training data using text-to-image diffusion models, but often require extensive model fine-tuning or external information sources. We present a novel training-free approach, called DIPSY, that leverages IP-Adapter for image-to-image translation to generate highly discriminative synthetic images using only the available few-shot examples. DIPSY introduces three key innovations: (1) an extended classifier-free guidance scheme that enables independent control over positive and negative image conditioning; (2) a class similarity-based sampling strategy that identifies effective contrastive examples; and (3) a simple yet effective pipeline that requires no model fine-tuning or external captioning and filtering. Experiments across ten benchmark datasets demonstrate that our approach achieves state-of-the-art or comparable performance, while eliminating the need for generative model adaptation or reliance on external tools for caption generation and image filtering. Our results highlight the effectiveness of leveraging dual image prompting with positive-negative guidance for generating class-discriminative features, particularly for fine-grained classification tasks.

---

### Repository Status
🚧 **Code coming soon!** The full codebase for DIPSY is being prepared for public release. Stay tuned for updates.
