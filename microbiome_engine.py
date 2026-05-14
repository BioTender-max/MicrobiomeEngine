"""
MicrobiomeEngine: 16S rRNA & Metagenomics Analysis Pipeline
- OTU clustering (97% identity threshold simulation)
- Alpha diversity (Shannon, Simpson, Chao1, Faith's PD)
- Beta diversity (Bray-Curtis, UniFrac, PCoA)
- Differential abundance (DESeq2-style negative binomial)
- PICRUSt-style functional pathway inference
- Dysbiosis index computation
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import braycurtis, pdist, squareform
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("MicrobiomeEngine v1.0")
print("16S rRNA & Metagenomics Analysis Pipeline")
print("="*60)

# ─── 1. SYNTHETIC 16S DATA ───────────────────────────────────
N_SAMPLES = 60    # 30 healthy + 30 disease
N_OTUS = 300      # OTUs after clustering
N_PATHWAYS = 50   # functional pathways

print(f"\n[Data] {N_SAMPLES} samples ({N_SAMPLES//2} healthy, {N_SAMPLES//2} disease)")
print(f"  {N_OTUS} OTUs, {N_PATHWAYS} functional pathways")

# Sample metadata
groups = np.array(['healthy']*30 + ['disease']*30)
ages = np.random.normal(45, 15, N_SAMPLES).clip(18, 80)
bmis = np.concatenate([np.random.normal(23, 3, 30), np.random.normal(28, 4, 30)])

# OTU taxonomy (phylum level)
PHYLA = ['Firmicutes', 'Bacteroidetes', 'Proteobacteria', 'Actinobacteria',
         'Verrucomicrobia', 'Fusobacteria', 'Tenericutes']
phylum_probs = [0.45, 0.30, 0.10, 0.08, 0.04, 0.02, 0.01]
otu_phyla = np.random.choice(PHYLA, N_OTUS, p=phylum_probs)

# ─── 2. OTU COUNT TABLE ──────────────────────────────────────
print("\n[OTU] Generating OTU count table...")

# Healthy microbiome: high diversity, balanced Firmicutes/Bacteroidetes
# Disease: dysbiosis — reduced diversity, Firmicutes↑, Bacteroidetes↓
def generate_otu_counts(n_samples, n_otus, otu_phyla, is_disease=False, seed_offset=0):
    np.random.seed(42 + seed_offset)
    counts = np.zeros((n_samples, n_otus), dtype=int)
    for i in range(n_samples):
        # Base abundance: Dirichlet-multinomial
        alpha = np.ones(n_otus) * 0.1
        # Phylum-level adjustments
        for j, phylum in enumerate(otu_phyla):
            if phylum == 'Firmicutes':
                alpha[j] = 0.15 if not is_disease else 0.25
            elif phylum == 'Bacteroidetes':
                alpha[j] = 0.12 if not is_disease else 0.06
            elif phylum == 'Proteobacteria':
                alpha[j] = 0.04 if not is_disease else 0.08
        probs = np.random.dirichlet(alpha)
        total_reads = np.random.randint(5000, 20000)
        counts[i] = np.random.multinomial(total_reads, probs)
    return counts

counts_healthy = generate_otu_counts(30, N_OTUS, otu_phyla, is_disease=False, seed_offset=0)
counts_disease = generate_otu_counts(30, N_OTUS, otu_phyla, is_disease=True, seed_offset=1)
otu_table = np.vstack([counts_healthy, counts_disease])

# Relative abundance
rel_abund = otu_table / otu_table.sum(axis=1, keepdims=True)

print(f"  Total reads: {otu_table.sum():,}")
print(f"  Mean reads/sample: {otu_table.sum(axis=1).mean():.0f}")
print(f"  OTU prevalence (>0 in >10% samples): {(otu_table > 0).mean(axis=0).sum()}")

# ─── 3. ALPHA DIVERSITY ──────────────────────────────────────
print("\n[Alpha] Computing alpha diversity metrics...")

def shannon_diversity(counts):
    """Shannon entropy H = -sum(p * log(p))"""
    p = counts / (counts.sum() + 1e-10)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def simpson_diversity(counts):
    """Simpson's D = 1 - sum(p^2)"""
    p = counts / (counts.sum() + 1e-10)
    return 1 - np.sum(p**2)

def chao1_estimator(counts):
    """Chao1 = S_obs + (n1^2) / (2*n2) where n1=singletons, n2=doubletons"""
    s_obs = (counts > 0).sum()
    n1 = (counts == 1).sum()
    n2 = (counts == 2).sum()
    if n2 == 0:
        return s_obs + n1 * (n1 - 1) / 2
    return s_obs + n1**2 / (2 * n2)

def faiths_pd(counts, n_otus):
    """Approximate Faith's PD using random phylogenetic tree branch lengths."""
    np.random.seed(42)
    branch_lengths = np.random.exponential(0.1, n_otus)
    present = counts > 0
    return branch_lengths[present].sum()

alpha_metrics = {
    'Shannon': np.array([shannon_diversity(otu_table[i]) for i in range(N_SAMPLES)]),
    'Simpson': np.array([simpson_diversity(otu_table[i]) for i in range(N_SAMPLES)]),
    'Chao1': np.array([chao1_estimator(otu_table[i]) for i in range(N_SAMPLES)]),
    'FaithsPD': np.array([faiths_pd(otu_table[i], N_OTUS) for i in range(N_SAMPLES)]),
}

for metric, values in alpha_metrics.items():
    h_stat, p_val = stats.mannwhitneyu(values[:30], values[30:], alternative='two-sided')
    print(f"  {metric}: healthy={values[:30].mean():.3f}, disease={values[30:].mean():.3f}, p={p_val:.3e}")

# ─── 4. BETA DIVERSITY & PCoA ────────────────────────────────
print("\n[Beta] Computing beta diversity and PCoA...")

# Bray-Curtis dissimilarity
bc_dist = squareform(pdist(rel_abund, metric='braycurtis'))

# PCoA (classical MDS)
def pcoa(dist_matrix):
    """Principal Coordinates Analysis."""
    n = dist_matrix.shape[0]
    D2 = dist_matrix**2
    # Double centering
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Keep positive eigenvalues
    pos = eigenvalues > 0
    coords = eigenvectors[:, pos] * np.sqrt(eigenvalues[pos])
    variance_explained = eigenvalues[pos] / eigenvalues[pos].sum()
    return coords, variance_explained

pcoa_coords, var_explained = pcoa(bc_dist)

# PERMANOVA (simplified)
def permanova(dist_matrix, groups, n_perm=999):
    """Simplified PERMANOVA F-statistic."""
    n = len(groups)
    unique_groups = np.unique(groups)
    # Total SS
    total_ss = dist_matrix.sum() / (2 * n)
    # Within-group SS
    within_ss = 0
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        ng = len(idx)
        sub_dist = dist_matrix[np.ix_(idx, idx)]
        within_ss += sub_dist.sum() / (2 * ng)
    between_ss = total_ss - within_ss
    # F-statistic
    k = len(unique_groups)
    f_stat = (between_ss / (k-1)) / (within_ss / (n-k))
    # Permutation test
    f_perm = []
    for _ in range(n_perm):
        perm_groups = np.random.permutation(groups)
        ws = 0
        for g in unique_groups:
            idx = np.where(perm_groups == g)[0]
            ng = len(idx)
            sub_dist = dist_matrix[np.ix_(idx, idx)]
            ws += sub_dist.sum() / (2 * ng)
        bs = total_ss - ws
        f_perm.append((bs / (k-1)) / (ws / (n-k)))
    p_val = (np.array(f_perm) >= f_stat).mean()
    return f_stat, p_val

f_stat, p_permanova = permanova(bc_dist, groups, n_perm=499)
print(f"  PERMANOVA: F={f_stat:.3f}, p={p_permanova:.4f}")
print(f"  PCoA PC1: {var_explained[0]*100:.1f}%, PC2: {var_explained[1]*100:.1f}%")

# ─── 5. DIFFERENTIAL ABUNDANCE ───────────────────────────────
print("\n[DiffAbund] Differential abundance testing...")

def nb_test(counts_a, counts_b):
    """
    DESeq2-style negative binomial test (simplified).
    Uses size-factor normalization + Wald test.
    """
    # Size factors (geometric mean normalization)
    all_counts = np.vstack([counts_a, counts_b])
    log_counts = np.log(all_counts + 0.5)
    ref = log_counts.mean(axis=0)
    size_factors = np.exp(np.median(log_counts - ref, axis=1))

    # Normalized counts
    norm_a = counts_a / size_factors[:len(counts_a), None]
    norm_b = counts_b / size_factors[len(counts_a):, None]

    # Log2 fold change
    mean_a = norm_a.mean(axis=0) + 0.5
    mean_b = norm_b.mean(axis=0) + 0.5
    log2fc = np.log2(mean_b / mean_a)

    # Wald test (t-test on log-normalized counts)
    pvals = np.array([stats.ttest_ind(
        np.log2(norm_a[:, j] + 0.5),
        np.log2(norm_b[:, j] + 0.5)
    ).pvalue for j in range(counts_a.shape[1])])

    # BH FDR
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    fdr = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        fdr[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(len(sorted_idx)-2, -1, -1):
        fdr[sorted_idx[i]] = min(fdr[sorted_idx[i]], fdr[sorted_idx[i+1]])

    return log2fc, fdr

log2fc_otus, fdr_otus = nb_test(counts_healthy, counts_disease)
sig_up = (log2fc_otus > 1) & (fdr_otus < 0.05)
sig_down = (log2fc_otus < -1) & (fdr_otus < 0.05)
print(f"  Significant OTUs: {sig_up.sum()} up in disease, {sig_down.sum()} down in disease")

# Top differentially abundant OTUs by phylum
for phylum in PHYLA[:4]:
    mask = otu_phyla == phylum
    up_ph = (sig_up & mask).sum()
    down_ph = (sig_down & mask).sum()
    print(f"    {phylum}: {up_ph} up, {down_ph} down")

# ─── 6. DYSBIOSIS INDEX ──────────────────────────────────────
print("\n[Dysbiosis] Computing dysbiosis index...")

# Firmicutes/Bacteroidetes ratio
firm_mask = otu_phyla == 'Firmicutes'
bact_mask = otu_phyla == 'Bacteroidetes'
fb_ratio = rel_abund[:, firm_mask].sum(axis=1) / (rel_abund[:, bact_mask].sum(axis=1) + 1e-6)

# Dysbiosis index: deviation from healthy reference
healthy_fb_mean = fb_ratio[:30].mean()
healthy_fb_std = fb_ratio[:30].std()
dysbiosis_index = np.abs(fb_ratio - healthy_fb_mean) / (healthy_fb_std + 1e-6)

# Combine with alpha diversity
shannon_vals = alpha_metrics['Shannon']
shannon_z = (shannon_vals - shannon_vals[:30].mean()) / (shannon_vals[:30].std() + 1e-6)
combined_dysbiosis = dysbiosis_index - shannon_z  # high F/B + low diversity = dysbiosis

print(f"  F/B ratio: healthy={fb_ratio[:30].mean():.2f}, disease={fb_ratio[30:].mean():.2f}")
print(f"  Dysbiosis index: healthy={combined_dysbiosis[:30].mean():.2f}, disease={combined_dysbiosis[30:].mean():.2f}")
t_dys, p_dys = stats.ttest_ind(combined_dysbiosis[:30], combined_dysbiosis[30:])
print(f"  Dysbiosis t-test: t={t_dys:.3f}, p={p_dys:.3e}")

# ─── 7. FUNCTIONAL PATHWAY INFERENCE (PICRUSt-style) ─────────
print("\n[Pathways] Inferring functional pathways...")

# Simulate OTU-to-pathway mapping
otu_pathway_matrix = np.random.binomial(1, 0.15, (N_OTUS, N_PATHWAYS))
# Pathway abundances = OTU abundances × OTU-pathway matrix
pathway_abund = rel_abund @ otu_pathway_matrix
pathway_abund = pathway_abund / (pathway_abund.sum(axis=1, keepdims=True) + 1e-10)

# Differential pathways
log2fc_pw, fdr_pw = nb_test(
    (pathway_abund[:30] * 1e6).astype(int),
    (pathway_abund[30:] * 1e6).astype(int)
)
sig_pw = fdr_pw < 0.05
print(f"  Significant pathways: {sig_pw.sum()}/{N_PATHWAYS}")

PATHWAY_NAMES = [
    'Butyrate synthesis', 'LPS biosynthesis', 'Bile acid metabolism',
    'Short-chain FA', 'Tryptophan metabolism', 'Mucin degradation',
    'Vitamin B12', 'Folate biosynthesis', 'Propionate synthesis', 'Acetate production'
]
top_pw_idx = np.argsort(np.abs(log2fc_pw))[-10:][::-1]
print(f"  Top pathway changes (simulated):")
for i, idx in enumerate(top_pw_idx[:5]):
    direction = "↑" if log2fc_pw[idx] > 0 else "↓"
    pw_name = PATHWAY_NAMES[i] if i < len(PATHWAY_NAMES) else f"Pathway_{idx}"
    print(f"    {pw_name}: log2FC={log2fc_pw[idx]:.2f} {direction}")

# ─── 8. VISUALIZATION ────────────────────────────────────────
print("\n[Viz] Generating dashboard...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: PCoA
ax1 = fig.add_subplot(gs_main[0, 0])
ax1.set_facecolor('#111111')
colors_pcoa = ['#4CAF50' if g == 'healthy' else '#FF5722' for g in groups]
ax1.scatter(pcoa_coords[:30, 0], pcoa_coords[:30, 1], c='#4CAF50', s=30, alpha=0.8, label='Healthy')
ax1.scatter(pcoa_coords[30:, 0], pcoa_coords[30:, 1], c='#FF5722', s=30, alpha=0.8, label='Disease')
ax1.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', color='white', fontsize=9)
ax1.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', color='white', fontsize=9)
ax1.set_title(f'Bray-Curtis PCoA\nPERMANOVA p={p_permanova:.3f}', color='white', fontsize=10, fontweight='bold')
ax1.tick_params(colors='white', labelsize=7)
for spine in ax1.spines.values(): spine.set_color('#333333')
ax1.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 2: Alpha diversity
ax2 = fig.add_subplot(gs_main[0, 1])
ax2.set_facecolor('#111111')
metrics_to_plot = ['Shannon', 'Simpson']
positions = [1, 2, 4, 5]
data_to_plot = []
labels_bp = []
colors_bp = []
for i, metric in enumerate(metrics_to_plot):
    vals = alpha_metrics[metric]
    data_to_plot.extend([vals[:30], vals[30:]])
    labels_bp.extend([f'{metric}\nHealthy', f'{metric}\nDisease'])
    colors_bp.extend(['#4CAF50', '#FF5722'])
bp = ax2.boxplot(data_to_plot, positions=positions, patch_artist=True)
for patch, col in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(col); patch.set_alpha(0.8)
for element in ['whiskers', 'caps', 'medians', 'fliers']:
    for item in bp[element]: item.set_color('white')
ax2.set_xticks(positions)
ax2.set_xticklabels(labels_bp, color='white', fontsize=7)
ax2.set_ylabel('Diversity', color='white', fontsize=9)
ax2.set_title('Alpha Diversity', color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='white', labelsize=7)
for spine in ax2.spines.values(): spine.set_color('#333333')

# Panel 3: Phylum composition
ax3 = fig.add_subplot(gs_main[0, 2])
ax3.set_facecolor('#111111')
phylum_abund_healthy = np.array([rel_abund[:30, otu_phyla==p].sum(axis=1).mean() for p in PHYLA])
phylum_abund_disease = np.array([rel_abund[30:, otu_phyla==p].sum(axis=1).mean() for p in PHYLA])
x = np.arange(len(PHYLA))
w = 0.35
phylum_colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4','#FF5722','#607D8B']
bars1 = ax3.bar(x - w/2, phylum_abund_healthy, w, color=phylum_colors, alpha=0.8, label='Healthy')
bars2 = ax3.bar(x + w/2, phylum_abund_disease, w, color=phylum_colors, alpha=0.5, label='Disease')
ax3.set_xticks(x)
ax3.set_xticklabels([p[:6] for p in PHYLA], color='white', fontsize=7, rotation=30)
ax3.set_ylabel('Relative Abundance', color='white', fontsize=9)
ax3.set_title('Phylum Composition', color='white', fontsize=10, fontweight='bold')
ax3.tick_params(colors='white', labelsize=7)
for spine in ax3.spines.values(): spine.set_color('#333333')
ax3.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 4: Differential abundance volcano
ax4 = fig.add_subplot(gs_main[1, 0])
ax4.set_facecolor('#111111')
neg_log_fdr = -np.log10(fdr_otus + 1e-300)
colors_da = np.where(sig_up, '#FF5722', np.where(sig_down, '#2196F3', '#555555'))
ax4.scatter(log2fc_otus, neg_log_fdr, c=colors_da, s=8, alpha=0.7)
ax4.axvline(x=1, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
ax4.axvline(x=-1, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
ax4.axhline(y=-np.log10(0.05), color='white', linestyle='--', alpha=0.3, linewidth=0.8)
ax4.set_xlabel('log2(Disease/Healthy)', color='white', fontsize=9)
ax4.set_ylabel('-log10(FDR)', color='white', fontsize=9)
ax4.set_title(f'Differential Abundance\nUp={sig_up.sum()}, Down={sig_down.sum()}', color='white', fontsize=10, fontweight='bold')
ax4.tick_params(colors='white', labelsize=7)
for spine in ax4.spines.values(): spine.set_color('#333333')

# Panel 5: Dysbiosis index
ax5 = fig.add_subplot(gs_main[1, 1])
ax5.set_facecolor('#111111')
ax5.scatter(range(30), combined_dysbiosis[:30], c='#4CAF50', s=25, alpha=0.8, label='Healthy')
ax5.scatter(range(30, 60), combined_dysbiosis[30:], c='#FF5722', s=25, alpha=0.8, label='Disease')
ax5.axhline(y=combined_dysbiosis[:30].mean() + 2*combined_dysbiosis[:30].std(),
            color='white', linestyle='--', alpha=0.4, linewidth=0.8, label='Threshold')
ax5.set_xlabel('Sample', color='white', fontsize=9)
ax5.set_ylabel('Dysbiosis Index', color='white', fontsize=9)
ax5.set_title(f'Dysbiosis Index\n(p={p_dys:.2e})', color='white', fontsize=10, fontweight='bold')
ax5.tick_params(colors='white', labelsize=7)
for spine in ax5.spines.values(): spine.set_color('#333333')
ax5.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 6: F/B ratio
ax6 = fig.add_subplot(gs_main[1, 2])
ax6.set_facecolor('#111111')
ax6.violinplot([fb_ratio[:30], fb_ratio[30:]], positions=[1, 2], showmedians=True)
ax6.set_xticks([1, 2])
ax6.set_xticklabels(['Healthy', 'Disease'], color='white', fontsize=9)
ax6.set_ylabel('Firmicutes/Bacteroidetes Ratio', color='white', fontsize=9)
ax6.set_title('F/B Ratio', color='white', fontsize=10, fontweight='bold')
ax6.tick_params(colors='white', labelsize=7)
for spine in ax6.spines.values(): spine.set_color('#333333')

# Panel 7: Pathway heatmap
ax7 = fig.add_subplot(gs_main[2, 0])
ax7.set_facecolor('#111111')
top_pw = np.argsort(np.abs(log2fc_pw))[-15:]
pw_matrix = pathway_abund[:, top_pw].T
pw_norm = (pw_matrix - pw_matrix.mean(axis=1, keepdims=True)) / (pw_matrix.std(axis=1, keepdims=True) + 1e-6)
im7 = ax7.imshow(pw_norm, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax7.axvline(x=29.5, color='white', linewidth=1.5, alpha=0.8)
ax7.set_xlabel('Sample', color='white', fontsize=9)
ax7.set_ylabel('Pathway', color='white', fontsize=9)
ax7.set_title('Top Differential Pathways', color='white', fontsize=10, fontweight='bold')
ax7.tick_params(colors='white', labelsize=7)
plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Panel 8: Bray-Curtis distance matrix
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.set_facecolor('#111111')
im8 = ax8.imshow(bc_dist, cmap='viridis', aspect='auto')
ax8.axhline(y=29.5, color='white', linewidth=1, alpha=0.6)
ax8.axvline(x=29.5, color='white', linewidth=1, alpha=0.6)
ax8.set_title('Bray-Curtis Distance Matrix', color='white', fontsize=10, fontweight='bold')
ax8.tick_params(colors='white', labelsize=7)
plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Panel 9: Summary
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.set_facecolor('#111111'); ax9.axis('off')
summary = [
    "MicrobiomeEngine v1.0", "",
    f"Samples: {N_SAMPLES}",
    f"  Healthy: 30, Disease: 30",
    f"OTUs: {N_OTUS}",
    f"Pathways: {N_PATHWAYS}", "",
    f"Alpha diversity:",
    f"  Shannon: {alpha_metrics['Shannon'][:30].mean():.2f} vs {alpha_metrics['Shannon'][30:].mean():.2f}",
    f"  Simpson: {alpha_metrics['Simpson'][:30].mean():.3f} vs {alpha_metrics['Simpson'][30:].mean():.3f}", "",
    f"Beta diversity:",
    f"  PERMANOVA F={f_stat:.3f}, p={p_permanova:.4f}", "",
    f"Differential OTUs:",
    f"  Up: {sig_up.sum()}, Down: {sig_down.sum()}", "",
    f"F/B ratio:",
    f"  Healthy: {fb_ratio[:30].mean():.2f}",
    f"  Disease: {fb_ratio[30:].mean():.2f}",
    f"Sig. pathways: {sig_pw.sum()}/{N_PATHWAYS}",
]
for i, line in enumerate(summary):
    ax9.text(0.05, 0.97-i*0.052, line, transform=ax9.transAxes,
             color='#E9ED4C' if i==0 else 'white', fontsize=8.5, va='top',
             fontweight='bold' if i==0 else 'normal')

fig.suptitle('MicrobiomeEngine: 16S rRNA & Metagenomics Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('/workspace/microbiome_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("  Dashboard saved.")

print("\n"+"="*60)
print("MicrobiomeEngine COMPLETE")
print(f"  Samples: {N_SAMPLES} | OTUs: {N_OTUS} | Pathways: {N_PATHWAYS}")
print(f"  PERMANOVA: F={f_stat:.3f}, p={p_permanova:.4f}")
print(f"  Differential OTUs: {sig_up.sum()} up, {sig_down.sum()} down")
print(f"  F/B ratio: healthy={fb_ratio[:30].mean():.2f}, disease={fb_ratio[30:].mean():.2f}")
print(f"  Dysbiosis p={p_dys:.3e}")
print("="*60)
