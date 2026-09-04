# 从 RIFT V3 到 V4：设计、实验与结论

> 状态快照：2026-09-03
> 目的：记录本仓库从原版 RIFT-SVC V3 出发所做的调查、实现、训练、失败尝试和当前结论。本文是技术决策记录，不是训练教程；当前可执行流程以 [training-protocol-v7.md](training-protocol-v7.md) 为准。

## 1. 项目目标与记录约定

V4 从一开始就没有把“复现原论文数字”作为目标。原版 checkpoint 无法反推出精确 WAV 清单、每位 speaker 的样本数、完整切分和所有预处理细节；继续追求历史兼容只会限制数据和模型设计。因此本项目采用以下原则：

- 质量优先，不保持 V1–V3 checkpoint、配置或数据格式兼容。
- 可获得、可审计的数据优先于猜测原训练集。
- 所有音频、特征、split、采样概率和 checkpoint 都必须能追溯。
- 主观试听最终优先，但结构指标和固定 panel 用来排查问题，不能只看单一 validation loss。
- 不因为一个改动“更标准”就默认它更好；保留受控比较和撤销路径。

本文使用三种状态：

- **保留**：当前代码或训练契约仍采用。
- **撤销/淘汰**：实际实现或训练过，但证据不足、风险过高或被后续方案替代。
- **未实现**：讨论或审计过，最终没有进入训练，不能当作实验结果。

## 2. 原版 V3 基线

### 2.1 从源码、commit 和 checkpoint 得到的事实

原版 V3 的核心路线是 rectified-flow/DiT 式 mel 生成器，输入 ContentVec、F0、RMS 和 speaker condition，输出 log-mel，再由冻结的 NSF-HiFiGAN 解码。我们审计的官方 `pretrain-v3_dit-1024-16.ckpt` 实际训练到 300,000 step，约 310.5M 参数。发布 checkpoint 含 119 个预训练 speaker 的映射信息，但故意未保存真人 `spk_embed.weight`；它保留了训练后的 null-speaker embedding。

原版单歌手后训练中：

| 部分 | 原版行为 |
|---|---|
| ContentVec | 预计算、冻结，不进优化器 |
| RMVPE/F0 | 预计算、冻结 |
| NSF-HiFiGAN | 只用于 validation/inference，不反传 |
| 主干 | Attention、FFN、输入/输出投影继续更新 |
| 默认冻结 | timestep embedding、各层 AdaLN projection、最终 AdaLN |
| 目标 speaker embedding | 重新随机初始化 |
| null speaker | 从预训练 checkpoint 继承 |
| speaker dropout | 单歌手默认 0 |
| 常见配置 | BF16、30k 上限、LR 5e-5、batch 64、5% warmup、256 frames |
| split | 文件级随机留出，不保证 song-disjoint |
| vocoder | 冻结的普通 NSF-HiFiGAN 2024.02 |

原代码对 null embedding 的“冻结”写法只设置了 module 属性，并没有冻结其 `weight`；因此 null embedding 在预训练的 speaker dropout 下实际学成了有效 unconditional condition，不能在迁移时简单置零。

### 2.2 V3 无法精确复现的部分

作者、issue 和 checkpoint 能确认使用过或考虑过 OpenSinger、GTSinger、M4Singer、Opencpop、KiSing/Kiritan、PopCS 等来源，但不能恢复：

- 每套语料具体保留了哪些 WAV；
- OpenSinger 为什么最后只保留 76 位 speaker；
- GTSinger 是否混入 paired speech；
- 长音频如何切分、去静音和去坏样本；
- 原始 train/test split 与每位 speaker 的精确曝光量。

因此 V4 不把官方 119-speaker mapping 当作训练数据真值，只把官方模型当作可比较的成熟基线。

### 2.3 Issue #11：真正被作者确认的 bug

[原版 issue #11](https://github.com/Pur1zumu/RIFT-SVC/issues/11) 指出发布代码把 attention scale 写成了 `1 / model_dim`。以 1024-wide、head_dim 64 为例，这实际是 `1/1024`。作者明确承认这里是 bug，并说明在原 spectral parameterization 下本意是 `1/head_dim`，即 `1/64`，而不是常规 Transformer 的 `1/sqrt(head_dim)`。

这一区分很重要：V4 不是把一个已确认应为 `1/sqrt(head_dim)` 的 typo 机械修正，而是做了两层决定：

1. 修复 `model_dim` 与 `head_dim` 混用的已确认 bug；
2. 同时放弃原 spectral parameterization，因此选择与普通 Xavier/AdamW/QK-Norm 配套的 `1/sqrt(head_dim)`。

作者在同一 issue 中只能回忆预训练使用了 GTSinger、Kiritan、KiSing、M4Singer、OpenSinger、Opencpop、PopCS，无法访问原预训练服务器。这正是 V4 转向独立数据审计而不是精确复现的直接起点。

## 3. 路线演进总览

| 阶段 | 主要尝试 | 结果 |
|---|---|---|
| V3 审计 | issue、commit、源码、官方 512×8/1024×16 checkpoint 与本地微调归档 | 明确继承边界；放弃精确复现 |
| 数据可得性与首轮准备 | GTSinger、M4Singer、OpenSinger、Opencpop、ACE、Kiritan；逐项下载/QC/特征 | 发现 GTSinger 漏收、ACE song key、重复音频和单 speaker 过采样问题 |
| Singing adapter pilot | 冻结 ContentVec，训练歌唱 adapter；尝试对抗式去 speaker | 漂移大、可归因证据不足、收益小；最终撤销 |
| 1024×12 pilot | 直接训练更大的 RIFT，沿用早期 duration sampling | 运行稳定，但数据与采样契约不可靠；约 70k 后作为 pilot 停止并清理 |
| 数据重建与 sampler V6 | 全量 GTSinger、song-disjoint、speaker/song bounded normalization | 保留，成为当前数据契约 |
| 1024×16 V6 | direct frozen ContentVec、QK-Norm、AdaLN-Zero、长窗口、EMA | 稳定训练到至少 150k，并形成完整结构轨迹 |
| V3/V4 公平 foundation 对照 | 同 8 首、同位置/noise、统一 raw log-mel | V4 150k 仍明显落后成熟 V3 300k |
| speaker pathway/leakage | correct/null/wrong、train/heldout、modulation、ContentVec probe | 显式 speaker 路径存在，但跨歌泛化弱；ContentVec 泄露强 speaker identity |
| 目标 singer 后训练 | 130k EMA、冻结 PC-NSF、song-disjoint 和 all-train 两条实验 | 训练可运行，但听感不如旧 V3；foundation 成熟度与 speaker leakage 成为首要嫌疑 |

## 4. 数据集调查、取舍与修复

### 4.1 最终纳入的数据

当前 canonical manifest 的完整数字见 [dataset-audit.md](dataset-audit.md)。摘要如下：

| Dataset | Accepted | Hours | Speakers | Songs | 当前角色 |
|---|---:|---:|---:|---:|---|
| OpenSinger | 42,947 | 51.900 | 76 | 1,127 | 主要真人多样性 |
| GTSinger | 26,711 | 75.291 | 20 | 610 | 主要真人、多语言、歌唱技巧 |
| M4Singer | 20,889 | 29.689 | 20 | 419 | 中文真人歌唱 |
| Opencpop | 3,752 | 5.218 | 1 | 100 | 低权重中文真人 anchor |
| Kiritan | 43 | 3.036 | 1 | 43 | 低权重单人补充 |
| ACE-Opencpop | 105,209 | 128.195 | 30 | 100 | 低权重合成增强，不进 PC-NSF |
| **总计** | **199,551** | **293.329** | **148 namespaced** | - | - |

训练 split 为 179,067 条、约 262.586 小时。mel statistics 与最终训练 manifest 都对应 81,348,986 个 train frames。

### 4.2 OpenSinger

最初主要障碍是 Google Drive 与国内云服务器链路。最终在个人电脑完成官方下载，再校验、解包并补传到服务器。审计确认原始 WAV 是 44.1 kHz，而不是此前推测的“部分原生 24 kHz”；因此不需要因 sample rate 排除它。

保留 76 位 speaker 不是 V4 人为裁剪，而是官方可获得数据经索引/QC 后形成的集合。V4 不试图猜测原版为什么也是 76，而是记录实际文件、哈希、拒绝原因和 split。

### 4.3 GTSinger

早期 pipeline 只索引 `control` 组：官方 singing WAV 为 28,628 条，但首轮 manifest 只索引 12,320、接受 11,451。这是足以改变训练分布的实质缺陷，触发了数据重建。

修复后纳入七个 singing group：control、breathy、falsetto、glissando、mixed voice、pharyngeal、vibrato，共接受 26,711 条。不同技术目录中存在 1,531 条 byte-identical 重复 control 音频，V4 只保留一份确定性副本。paired speech 没有纳入，因为它改变任务分布，且没有足够证据证明收益大于风险。

### 4.4 Opencpop 与 ACE-Opencpop

Opencpop 是单一真人歌手，不是合成语料。ACE-Opencpop 才是合成扩展。两者共享相同的 100 个 source-song identity，必须共享 split namespace，否则合成版本与真人版本会跨 train/validation 泄漏同一首歌。

ACE release 有 105,960 个 parquet row，其中 510 个是 zero-frame WAV；canonical raw manifest 正确包含其余 105,450 条。早期使用 `acesinger_N` 作为 song key 会破坏 song-disjoint，已改成真实 source song。

### 4.5 Kiritan、KiSing、ACE-KiSing、PopCS 与其他语料

- Kiritan 质量和规模一般，只有一位 singer；最终保留但固定为 1% 低权重。
- KiSing/ACE-KiSing 对当前数据覆盖的增量有限，质量与域一致性风险较高，未纳入最终 manifest。
- PopCS 没有纳入。
- CCMUSIC、SingingVoiceDataset、OpenCPOP 等候选经过讨论，但没有证据表明加入后一定优于当前混合；它们未进入本轮训练，不能把“未使用”解释为已证实质量差。

### 4.6 音频 QC 与特征完整性

V4 将 duration、mostly-silent、near-silent、clipping、duplicate audio 等拒绝原因写进 manifest，不再依赖静默删除。所有 accepted 条目必须同时存在：

- 与 PC-NSF 契约一致的 128-bin log-mel；
- F0/voicing；
- RMS；
- pinned dual-phase ContentVec。

特征完成后以实际 mel frame length 回写 manifest，避免 WAV duration、F0、mel 和 sampler 看到不同帧数。

## 5. ContentVec 与条件特征

### 5.1 双相 ContentVec 与时间对齐

V3 使用了将原音频与约 10 ms shift 特征交织的双相 ContentVec 思路。V4 继承了“双相”本身，但修正了 shift/interleave 的时间对齐：两相特征先映射到同一个 mel-frame 时间轴，再组合；不再用会产生半帧错位的直接交叉拼接。

这项做法不是标准 ContentVec API，也没有被当作已由论文充分证明的通用技巧。保留它的理由是：它能在不更换 encoder 的情况下补充 20 ms ContentVec stride 下的中间时间相位，并且原项目已依赖这一信号。风险是额外计算、边界处理复杂和潜在 speaker leakage，因此现在已有独立审计脚本。

### 5.2 Singing adapter pilot（已撤销）

首轮方案是冻结预计算 ContentVec，再训练歌唱 adapter，希望：

- 适配歌唱发音、长元音和高音；
- 保留冻结 encoder 的稳定性；
- 未来可以单独审计 adapter，而不重训 ContentVec。

训练中补加了 loss/gradient finite 检查、adapter LR 和 drift 记录。回溯审计看到某些阶段整体参数漂移约 11%–25%，最大单 tensor 曾达到约 45.8%；而可测得的交叉验证收益只有约 1.22%。旧策略 `keep_last_checkpoints=3` 又删除了部分早期 adapter，使“漂移是正常适配还是退化”难以严格归因。

尝试过对抗式 speaker removal，但训练/IO 成本、评价困难和不确定收益不匹配。最终决定：

- 撤销对抗训练 Singing adapter；
- 当前 1024×16 run 直接使用冻结的双相 ContentVec；
- adapter 相关 checkpoint 只作为历史证据，不作为当前 foundation 依赖。

### 5.3 第二语义流及其他 encoder（未实现）

讨论过 WavLM-Large、SPIN、Xeus、Hybrid Multi-Feature Fusion 和第二语义流。它们可能改善内容鲁棒性，但会显著增加对齐、显存、特征缓存和归因复杂度。在当前单一 ContentVec 路径尚未证明优于 V3 前，引入第二流的收益/风险比不合适，因此只做接口/文献审计，没有进入实现。

## 6. Mel 与 vocoder 契约

### 6.1 当前 mel 契约

V4 固定为 44.1 kHz、hop 512、128 bins、FFT/window 2048、Slaney mel、40–16,000 Hz、自然对数、`center=false`，并使用每个 mel channel 的 dataset mean/std。

这样做的动机是直接匹配 OpenVPI 官方 PC-NSF-HiFiGAN 2025.02 checkpoint，避免用一个 mel 定义训练 RIFT、再用另一个定义解码。

### 6.2 PC-NSF 接入

固定了 OpenVPI SingingVocoders 源码 revision、官方 archive、checkpoint 哈希和 mel contract。PC-NSF 在 RIFT 训练中完全冻结，只在 validation/inference 把预测 mel + F0 解码成 waveform。RIFT 训练本身不经过 vocoder，也不会从 vocoder 反传。

早期发现 PC-NSF prepare 会把 manifest 中所有 accepted 音频写入训练列表，包括不适合 vocoder 的合成 ACE。这个缺陷**没有影响 RIFT flow 训练**，因为当时没有训练 PC-NSF；它会污染未来 vocoder 自训，因此当前数据配置明确排除 ACE。

本轮没有自训或微调 PC-NSF。原因是先验证官方 checkpoint 是否足够；如果 mel-domain foundation 本身落后，先训练 vocoder无法解决 RIFT vector field 的问题。

### 6.3 BigVGAN/APNet 等（未实现）

讨论过 BigVGAN-v2、MS-ISTFT-HiFiGAN、APNet 和 diffusion vocoder。BigVGAN 不显式输入 F0 可能带来更自然的 waveform 建模，也可能在歌唱音高稳定性和跨音域泛化上不如显式 F0 vocoder。由于要公平使用 BigVGAN，RIFT 和 vocoder 必须共享严格 mel 契约，并需要单独训练/验证 vocoder；本轮没有执行，当前仍以 PC-NSF 为唯一 pinned decoder。

## 7. 模型参数化的变化

### 7.1 Attention scale：`1/head_dim` → `1/sqrt(head_dim)`

发布实现误用了 `1/model_dim`，而作者说明 spectral parameterization 下本意是 `1/head_dim`。我们仍认真考虑过“不带 sqrt”是否是作者故意压平 softmax、配合 spectral parameterization 和特殊 optimizer 的整体设计，而不是应该孤立改掉的公式。

最终仍改为标准 `1/sqrt(head_dim)`，理由是：

- head dimension 固定为 64 后，512×8、768×12、1024×16 的 attention statistics 可以保持一致；
- 更容易与 PyTorch fused SDPA、QK-Norm 和常规初始化共同解释；
- 避免宽度变化同时改变 attention temperature；
- 原版的特殊 scale 与 spectral initialization/fan-ratio optimizer 强耦合，而 V4 已放弃整套参数化。

为控制风险，V4 加入逐层 attention entropy、logits std、max probability、Q/K norm 和 residual 审计。当前 QK-Norm 是 `elementwise_affine=false` 的参数无关 LayerNorm，因此没有可训练 gamma；早期“监控 gamma 判断温度”的设想不适用于最终实现。

### 7.2 撤销 spectral initialization 与 fan-ratio optimizer

原版用 spectral initialization 和按 fan ratio 缩放优化器，可能是在补偿非标准 attention scale 和不同层宽度的统计差异。V4 将其作为一套整体撤销，而不是只删其中一项：

- Linear 使用 Xavier uniform；Embedding 使用 std 0.02 normal；
- AdaLN modulation 和 output 保持 zero-init；
- 全模型使用普通 AdamW、统一 base LR；
- 梯度裁剪与 finite checks 负责运行时保护。

这样牺牲了一部分原作者可能有意设计的隐式缩放，但换来了 width/depth 可解释性、BF16 稳定性和更容易复现实验的优化器语义。

### 7.3 QK-Norm、RoPE、Pre-Norm 与 AdaLN-Zero

当前 block 使用：

- parameter-free QK-Norm；
- RoPE 时间位置编码；
- Pre-Norm residual stream；
- AdaLN-Zero 的 attention/FFN shift、scale、gate；
- depthwise-convolution gated FFN；
- zero-init final modulation 与输出层。

AdaLN-Zero 不是只靠“初始稳定”来证明合理。审计持续记录 gate RMS、绝对 residual RMS、residual/input ratio、activation RMS 和 alignment cosine，确认 16 层从零有序打开，而不是长期关闭或瞬间爆炸。

### 7.4 1024×12 pilot → 1024×16

早期 V4 pilot 采用 1024×12，希望以较少层数验证更长 context、数据 pipeline 和新参数化。它运行稳定，但后期层负载表明深层 FFN 和最后 attention 仍承担较强 residual；同时官方成熟 V3 有 1024×16 版本。下一轮在保持 width=1024、head_dim=64、FFN ratio、初始化和优化器不变的情况下增加到 16 层。

16 层不是因为“层数越多一定越好”，而是为了让容量扩展保持 head statistics 不变，并把 12 层观察到的后端负载分散到更多 block。12,288-frame GPU smoke test 通过，正式训练实测峰值显存约 17.3 GiB，典型吞吐约 2.0–2.2 step/s、20k–21k frames/s；32 GiB GPU 足够。

### 7.5 推理 schedule

V4 加入 cosine time schedule 和 Heun solver，作为质量优先的默认推理路径。后来的 V3/V4 公平比较同时跑了统一的 32-interval linear Euler 和各自 native solver；结论一致，因此 V4 落后不能归因于 Heun/cosine 设置。

## 8. 采样与 split 的重构

### 8.1 早期 duration sampling 的问题

按 dataset duration 或 song duration 直接抽样看似“充分利用数据”，但对单 speaker dataset 会造成极端 physical-speaker exposure。例如 Opencpop 即使只有约 2% dataset mass，其唯一真人 singer 的训练概率仍可能是普通 GTSinger singer 的十几倍。Kiritan也有同类风险。

这与后期观察到的单人语料 validation 行为不稳定相符，因此 sampler 从 dataset 层继续审计到 physical speaker 与 song。

### 8.2 当前 bounded normalization

在每个 dataset 内，以 `sqrt(accepted_frames)` 作为温和 duration prior，再用 capped-simplex/water-filling 投影：

```text
sum(q_s) = 1
0.5 / N <= q_s <= 2.0 / N
```

song 层采用同样的 `sqrt(hours)` 与上下界；最后才在 song 内按可用 frames 选 recording。ACE-Opencpop 与 Opencpop 合并为一个 source family，合计上限 3.5%。单一真人 speaker 的全局概率不得超过真人 speaker 中位数的三倍。

早期 duration sampler 出现的极小浮点归一化误差已通过显式 capped-simplex projection 和最终质量守恒检查解决，不再靠容差掩盖。

### 8.3 Song-disjoint validation

V4 split key 是原曲/song identity，不是切片文件。整首原曲可以切成很多训练 crop，但同一原曲的所有片段只能出现在一个 split。这样既充分利用训练音频，又避免同歌不同片段泄漏到 validation。

目标 singer 数据同样按 19 首原曲切分。另做了一条 all-train 实验，用于在不依赖不稳定小验证集的情况下最大化目标歌手数据；该实验不能用自身 validation loss 选择 checkpoint，只能靠固定外部歌曲和试听。

## 9. 训练系统与可观测性的变化

### 9.1 优化和安全检查

当前 foundation 配置：AdamW 2e-4、weight decay 0.01、10k warmup、cosine 到 2e-5、BF16、clip 1.0、EMA 0.9999、12,288-frame batch budget、256/384/512 frame buckets。

训练循环新增：

- total loss、LR、frames/s、samples/s、GPU memory；
- clip 前 gradient norm 与 non-finite 检查；
- loss/gradient/model/EMA/optimizer 任一非有限立即终止；
- 坏状态不写 checkpoint；
- resume 时严格验证 schema、config、manifest hash、mel statistics 和 speaker mapping；
- 8 个 worker 的数据加载；当前实测不是显存瓶颈。

10k warmup 曾被质疑可能与 AdaLN-Zero、zero output 重复保守，但没有在本轮中途修改。它保留为 baseline，未来只有在 gate/output 长期打不开时才做 5k/2k 对照。

### 9.2 Checkpoint 策略

早期只保留最近 3 个 checkpoint，导致 adapter drift 无法完整回溯。当前改为：

- 每 2k：model + EMA + config 的轻量 audit checkpoint；
- 每 5k：含 optimizer、RNG、epoch/batch offset 的 full resume checkpoint；
- validation 改善：`best.pt`；
- 结束：full `final.pt`；
- checkpoint index 记录类型、step 和路径。

硬盘空间足够，因此不再自动清理有诊断价值的历史 checkpoint。

### 9.3 Validation 与结构审计

原来只看 aggregate loss 容易被简单 timestep、单人 dataset 或大量合成条目主导。V6 validation 改为：

- 每首歌先聚合 recording；
- 再聚合到 speaker；
- 再形成 real-speaker macro、real-dataset macro、all-dataset macro 和 mixture-weighted loss；
- 主 selection metric 为 EMA `real_speaker_macro_flow`；
- synthetic validation 不选择 `best.pt`；
- 固定 panel/noise，保留 per-song/per-speaker JSON。

新增的结构审计包括：

- RF validation loss、MSE/cosine、velocity RMS 按 t 分桶；
- attention entropy/logits std/max probability；
- Q/K norm；
- AdaLN attention/FFN gate、shift、scale RMS；
- absolute attention/FFN residual RMS；
- residual/input ratio、pre/post stream RMS、alignment cosine；
- 每层 activation RMS；
- F0/voicing 分布与固定 audio panel；
- online 与 EMA 的固定 checkpoint 对照。

## 10. 实际训练与实验结果

### 10.1 Adapter 与 1024×12 pilots

Adapter pilot 暴露了“没有持续保存结构状态就无法解释 drift”的问题；1024×12 pilot 则暴露了数据清单和采样问题。两者的主要价值是完善 pipeline，而不是产生可发布模型。

1024×12 曾观察到约 0.14619 的 EMA validation，但它使用不同数据量、sampler、validation 聚合和模型深度，不能直接与 V6 数字比较。数据重建后停止并清理约 70k 的 pilot，不归档为候选 foundation。

### 10.2 1024×16 V6 结构行为

1024×16 从零训练时没有 NaN、持续撞 clip、activation 爆炸或 attention 全部 one-hot/uniform 的证据。AdaLN gates 从零打开并形成 layer-specific 分布。最后一层 attention residual 随训练增大，因此在 130k 和 150k 分别做了 L15 residual scale sweep；缩放最后一层没有给出足以改模型结构的稳定优势，L15 不是整体质量差距的主要原因。

逐层审计覆盖 2k、10k、20k、30k、35k、50k、60k、65k、70k、80k、90k、100k、110k、120k、125k、130k、135k、138k、140k 和 150k 附近。结论是 residual 增长总体平滑，不是某一步突然失控；high-t 区域学习较难，但 V3 也有相同趋势。

### 10.3 V4 checkpoint trajectory

使用相同 8 首、相同位置、固定 Gaussian noise、null speaker、EMA 和 32-interval linear Euler，把预测统一反归一化到 raw log-mel：

| Step | Raw MSE ↓ | Raw L1 ↓ | Cosine ↑ | Common standardized MSE ↓ |
|---:|---:|---:|---:|---:|
| 40k | 2.2566 | 1.1483 | .96468 | .43410 |
| 65k | 2.0927 | 1.1074 | .96779 | .40399 |
| 80k | 2.0223 | 1.0922 | .96814 | .39175 |
| 100k | 2.0180 | 1.0919 | .96780 | .38959 |
| 120k | 1.8933 | 1.0536 | .96919 | .36517 |
| 130k | 1.9648 | 1.0750 | .96803 | .37794 |
| 140k | 1.9509 | 1.0742 | .96837 | .37640 |
| 150k | **1.8456** | **1.0431** | **.97004** | **.35685** |
| V3 300k | **1.1906** | **.8188** | **.97510** | **.23480** |

这条轨迹推翻了“130k 已是绝对上限”的早期判断：V4 到 150k 仍能改善，原 real-speaker validation 的局部平台不能代表 foundation 成熟度。按 frame exposure，V3 约见过 4.915B frames，V4 130k 约 1.597B，只是约三分之一；参数量还从约 310.5M 增到约 377.1M。

当前 Hugging Face 上的 130k EMA 是在后续 150k trajectory 和 speaker-leakage 审计完成前发布的可复现实验快照。它不应被解读为“已证明最优的 V4 foundation”；150k 也没有仅凭单个 panel 自动替换发布模型。

但轨迹也表明：训练不足不是唯一解释。150k 的 raw/common-standardized 指标都仍明显落后 V3，不能只靠“多跑到 500k”保证解决。

### 10.4 Per-bin normalization/objective 审计

V3 使用固定 affine normalization：

```text
z = (raw_logmel + 12) / 7 - 1
```

V4 使用 channelwise standardization：

```text
z_c = (raw_logmel_c - mean_c) / sigma_c
```

因此 V4 的未加权 normalized MSE 等价于 raw error 按 `1/sigma_c²` 加权。我们用 V4 的 `sigma_c` 同时重算 V3/V4 common-standardized metric，并比较 `sigma_c^alpha`，`alpha=0/1/2`。trajectory 中 raw metric 和 standardized metric 方向一致，没有出现“standardized 持续改善而 raw 持续恶化”的强证据。

150k 时 128 个 bin 中 V4 在 118 个 bin 更差、10 个更好；V4/V3 raw error ratio 与 sigma 的 Spearman rho 约 0.484，线性 log-log `R²≈0.139`。这说明 normalization geometry 可能贡献了一部分频带差异，但解释力有限，不能把整体差距归咎于 mel normalization，也不足以支持立即把 loss 改成 `sigma²` 权重。

### 10.5 V3/V4 公平 foundation 对照

控制条件：同 8 首目标歌手歌曲、同物理位置、256 frames、相同 noise seed、双方各自原生特征/normalization，统一 32-step Euler，并反归一化到 raw log-mel。

| Metric | V3 300k | V4 130k |
|---|---:|---:|
| Raw MSE | **1.191** | 1.965 |
| Raw L1 | **.819** | 1.075 |
| Raw cosine | **.9751** | .9680 |

V3 在 8/8 歌上 MSE 更好，paired bootstrap 区间不跨 0。用双方 native solver 后结论不变：V3 Euler linear MSE 1.187，V4 Heun cosine MSE 2.005。

`.10–.95` 的 velocity curve 基本都由 V3 占优，因此差距不是只发生在 high-t，也不是 solver 调度造成。

### 10.6 Correct/null/wrong speaker 与 train/heldout

由于官方 V3 没保存真人 speaker embedding，无法运行严格的 V3-correct。对 V4 使用 8 位共同真人 speaker、每人两首/crop、固定 wrong-speaker rotation：

| Panel / condition | Raw MSE ↓ | Raw L1 ↓ | Cosine ↑ |
|---|---:|---:|---:|
| Heldout · V3 null | **.6269** | **.6029** | **.9901** |
| Heldout · V4 correct | 1.0209 | .7418 | .9836 |
| Heldout · V4 null | 1.0138 | .7579 | .9838 |
| Heldout · V4 wrong | 1.2136 | .8204 | .9813 |
| Train · V3 null | **.6444** | **.6106** | **.9894** |
| Train · V4 correct | .8420 | .6654 | .9828 |
| Train · V4 null | 1.0602 | .7554 | .9800 |
| Train · V4 wrong | 1.1975 | .8287 | .9783 |

Train crop 上 correct 比 null 的 MSE 好 20.6%，比 wrong 好 29.7%；heldout 上 correct 与 null 几乎持平，但仍比 wrong 好 15.9%。所以 speaker path 确实工作，却主要在训练歌曲上产生收益，跨歌泛化不足。

`correct-null` 对 velocity output 的相对影响约 3.5%–6%，`correct-wrong` 约 6%–12%，并随 t 增大而减弱。该现象与 high-t 时 time-conditioned absolute modulation 增大、speaker delta 相对占比下降一致。

### 10.7 Speaker condition 注入审计

Correct speaker embedding RMS 约 .0290，null 约 .0249，time embedding 约 .63–.74。speaker embedding 虽小，但 modulation projection 会放大它。以 `t=.5` 的 16 层均值为例：

| Branch | Correct absolute RMS | Correct-null delta RMS | Delta / absolute |
|---|---:|---:|---:|
| Attention shift | .0642 | .0182 | 30.5% |
| Attention scale | .7392 | .0506 | 7.3% |
| Attention gate | .1678 | .0338 | 21.4% |
| FFN shift | .0891 | .0315 | 36.3% |
| FFN scale | .4843 | .0412 | 9.1% |
| FFN gate | .3726 | .0384 | 11.0% |

16 层全部存在非零 correct/null modulation difference。因此问题不是 speaker embedding 没接入、AdaLN 没响应或实现漏传 condition。

### 10.8 ContentVec speaker leakage

使用冻结双相 ContentVec 的 temporal mean+std（1536D）训练简单 logistic-regression probe，训练歌曲拟合，在严格 song-disjoint heldout songs 上识别 speaker：

| Probe | Chance | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| 8 speakers | 12.5% | **92.11%** | **91.70%** |
| GTSinger 4 speakers | 25% | **98.35%** | **98.45%** |
| M4Singer 4 speakers | 25% | **86.36%** | **85.77%** |

按整首歌聚合后的 8-speaker accuracy 为 98.08%。GTSinger 可能混有语言/录音条件信息，但同一数据集内部的 M4Singer 仍有 85.8%，足以证明 ContentVec 携带很强的 singer identity。

这解释了为什么 null condition 仍能重建、heldout correct/null 很接近：预训练目标允许模型绕过显式 target speaker，从 content 中恢复 source identity。它也是目前不建议原样续训到 300k/500k 的主要原因。

## 11. 目标歌手后训练

### 11.1 V3 历史对照

本地归档保存过同一目标歌手的 V3 微调：

- 768×12：保留 7.5k/22.5k 等权重；
- 1024×16：保留 25k inference weight 与 40k full resume；
- 19 个原始文件切成 472 个 train audio entry；
- 旧评估曾用单个约 2.97 秒固定验证片段，并记录 MCD、SI-SNR、PSNR、MSE。

旧指标适合在同一 V3 run 内排 checkpoint，但不能跨 V3/V4 直接比较；小 panel 容易偶然，却提供了“固定输入比随机 validation 更稳定”的经验。

### 11.2 V4 fine-tune recipe

从 130k EMA foundation 初始化：

- 目标 speaker embedding 使用 foundation 真人 embedding 均值，而非随机；
- 继承非零 null embedding；
- 冻结 timestep embedding、16 层 modulation 和 final modulation；
- attention、FFN、输入/输出投影继续训练；
- backbone LR 1e-5，speaker LR 5e-5；
- 15k 上限、1k warmup、EMA .999；
- 冻结官方 PC-NSF；
- 保持 foundation mel statistics。

先做 song-disjoint run，早期 validation 为：1k .24133、2k .23243、3k .23874、4k .26012。19 首数据的 validation 方差和训练数据损失都较明显，单靠这条曲线无法稳定选模型。随后完成 all-train 15k run，并导出两首外部歌曲试听。

主观结果不如旧 V3 微调。这个结果不能只归因于 PC-NSF，因为 RIFT 训练不经过 vocoder，而且公平 raw-mel foundation audit 已经显示 V4 底模落后。最新 speaker-leakage 结果进一步说明：目标歌手后训练可能同时受 foundation 未成熟和 content source-identity 捷径影响。

## 12. 已保留、已撤销和未实现的决策

### 12.1 当前保留

- 44.1 kHz / hop 512 / 128-bin PC-NSF mel contract。
- 官方冻结 PC-NSF 2025.02；暂不自训 vocoder。
- 冻结、预计算、已校正时间对齐的双相 ContentVec。
- 显式 F0/voicing 与 RMS。
- 1024×16、head_dim 64、QK-Norm、RoPE、Pre-Norm、AdaLN-Zero。
- 标准 `1/sqrt(head_dim)`、Xavier/zero-init、普通 AdamW。
- 256/384/512 context bucket 与 12,288-frame budget。
- bounded speaker/song sampler 与 source-family cap。
- song-disjoint split、EMA、固定 panel/noise、结构审计和高保留 checkpoint。
- cosine schedule + Heun 作为默认推理，同时保留 linear Euler 审计能力。

### 12.2 已撤销或淘汰

- 追求 V3 精确数据复现与 checkpoint 兼容。
- 对抗训练 Singing adapter。
- 依赖 adapter 自由漂移到 300k 的方案。
- 1024×12 pilot 作为最终 architecture。
- 只按 duration/dataset 采样、不约束 physical speaker/song exposure。
- 文件级随机 validation split。
- `keep_last_checkpoints=3` 式激进清理。
- 用单一 aggregate validation loss 或单个 2.97 秒片段决定 foundation。
- 认为 130k 已必然收敛，或认为 L15 是主要质量问题。

### 12.3 讨论过但未实现

- 第二 semantic stream、WavLM/SPIN/Xeus/hybrid fusion。
- BigVGAN-v2、APNet、MS-ISTFT、diffusion vocoder。
- 自训/微调 PC-NSF。
- GTSinger paired speech。
- KiSing/ACE-KiSing、PopCS、CCMUSIC、SingingVoiceDataset 的正式加入。
- 将 loss 直接改成 `clamp(sigma²)` 或固定 `sigma^alpha` weighting。
- 对 ContentVec 做 speaker-adversarial adapter、实例归一化或 wrong-speaker margin loss。

## 13. 当前判断与下一步门槛

目前不能把 V4 的问题简化为“没训够”或“参数化改坏了”中的任一项。证据同时支持：

1. **训练预算不足**：150k trajectory 仍改善，frame/parameter exposure 远低于成熟 V3。
2. **显式 speaker 条件泛化不足**：train correct 明显优于 null，heldout correct/null 却接近。
3. **ContentVec 存在强 speaker leakage**：预训练 reconstruction 可以走 source-identity 捷径。
4. **mel normalization 可能影响频带权重，但不是主要单因**：raw/common-standardized trajectory 同向，per-bin 回归解释力有限。
5. **结构没有明显数值崩坏**：attention、AdaLN、residual、activation 与 L15 sweep 没有找到能解释整体差距的单点故障。

因此截至本文快照，**不建议把当前 recipe 原样长跑到 300k/500k**。推荐先从同一 checkpoint 做短受控分支：

- unchanged control；
- 温和 ContentVec de-leak 方案，例如 utterance-level temporal centering/受限 normalization；
- 可选低概率、同数据集 wrong-speaker ranking，用于迫使 correct condition 比 wrong 更接近 target；
- 同时复跑 heldout correct/null/wrong、ContentVec probe、raw foundation panel 与实际转换试听。

只有当修正分支在不损害 phonetic/content reconstruction 的前提下，稳定扩大 heldout correct-null/correct-wrong separation，并改善目标歌手试听，才值得用新 recipe 从零开始正式长训练。短分支本身只用于选择方向，不应直接包装为最终模型。

## 14. 证据与复现入口

仓库内：

- 数据总表：[dataset-audit.md](dataset-audit.md)
- 数据来源：[dataset-availability.md](dataset-availability.md)
- 数据布局：[dataset-layout.md](dataset-layout.md)
- 当前训练契约：[training-protocol-v7.md](training-protocol-v7.md)
- foundation trajectory：[audit_foundation_trajectory.py](../scripts/audit_foundation_trajectory.py)
- known-speaker 对照：[audit_known_speaker_conditioning.py](../scripts/audit_known_speaker_conditioning.py)
- speaker pathway/modulation：[audit_speaker_pathways.py](../scripts/audit_speaker_pathways.py)
- ContentVec leakage：[audit_content_speaker_leakage.py](../scripts/audit_content_speaker_leakage.py)
- PC-NSF 固定资产：[pc_nsf_hifigan.lock.json](../third_party/pc_nsf_hifigan.lock.json)

服务器训练快照中的关键记录文件：

```text
records/foundation-trajectory-objective-analysis.json
records/foundation-native-common-panel.json
records/v3-foundation-300k-per-bin.json
records/v4-foundation-trajectory-per-bin.json
records/speaker-pathways-train-heldout.json
records/contentvec-speaker-leakage.json
records/rift-v6-*-structure-audit-cpu.json
records/rift-v6-*-pre-gate-audit-cpu.json
records/rift-v6-*-l15-residual-sweep*.json
runs/rift/run_metadata.json
runs/rift/run_events.jsonl
runs/rift/sampling-audit.json
```

这些大文件不全部提交到 Git；脚本、协议、摘要数字和文件名共同构成可复核的 provenance。任何后续文档若与这些固定 JSON 冲突，应以带 checkpoint hash、manifest hash 和 config 的原始审计记录为准，并更新本文而不是静默覆盖历史结论。

## 15. V7 clean-run architecture and optimization reset

The prior trajectory and LR branches were discarded as pilots. The next run
starts from step 0 with a constant post-warmup LR, lower speaker dropout,
explicit optimizer decay groups, and correct-speaker endpoint ranking as
defined in `training-protocol-v7.md`.

Before launching on the RTX 5090, the gated convolutional FFN width changed
from 4,096 to 2,816. This preserves the three-matrix gated topology while
making its Linear parameter budget approximately equal to a conventional 4x
two-matrix FFN. The exact 148-speaker model size is 313,500,800 parameters,
down from 377,081,984. Gate/up and QKV projections were already fused in the
source; no new mathematical approximation was introduced for those fusions.

The formal performance candidate uses PyTorch 2.12.1, CUDA 13.0, Python 3.12,
TorchAO 0.18.0 rowwise FP8 with high-precision weight gradients, Flash-only
SDPA, `torch.compile`, fused AdamW, and TF32 for residual FP32 GEMM/conv work.
FP8 is restricted to the 64 attention and FFN Linear modules. The conversion
must preserve canonical state keys, and release inference remains ordinary
high-precision Linear. The target-singer fine-tune stays BF16 rather than
inheriting FP8 automatically.

This stack is conditional on the 5090 smoke and performance gates. A source
configuration saying `float8_training=true` is not evidence of acceleration:
the formal run may start only after an actual forward/backward/optimizer step
succeeds with Flash fallback disabled and compiled FP8 beats or credibly
matches compiled BF16 on the fixed 16,384-frame benchmark.

## 16. V7 sealed shadow panel

The original 16-sample endpoint panel showed substantial gains through 220k,
but was too small to establish generalization. A new panel was therefore
sealed before examining its checkpoint results. Selection is deterministic and
contains no manually chosen songs. The lock hashes the complete source
manifest, mel statistics, speaker mapping, every selected feature tensor, crop
position, and protocol parameters.

The resulting panel contains 128 crops from 116 independent song units and 60
physical speakers: GTSinger 68, M4Singer 25, OpenSinger 27, and Kiritan 8. Each
512-frame crop is the prefix of the corresponding locked 768-frame crop, and
all checkpoints receive the same Gaussian noise. Evaluation uses EMA weights,
correct speaker conditioning, and 32-step linear Euler integration.

| step | 512 active mean | 512 active median | 768 active mean | 768 active median |
|---:|---:|---:|---:|---:|
| 130k | 1.06836 | 0.86121 | 0.96386 | 0.77228 |
| 170k | 0.97836 | 0.75085 | 0.89886 | 0.72366 |
| 200k | 0.94500 | 0.79367 | 0.84183 | 0.67533 |
| 220k | 0.91639 | 0.73442 | 0.82347 | 0.63032 |

The 130k to 220k improvement is robust: active-MSE song-bootstrap intervals do
not cross zero at either length. The narrower 200k to 220k comparison is not
statistically decisive. Its active paired deltas are -0.02861 at 512 frames
(median -0.01231, 68.0% wins, 95% song-bootstrap CI [-0.09174, 0.05043]) and
-0.01836 at 768 frames (median -0.00814, 59.8% wins, CI [-0.05997, 0.04467]).

Full-MSE means slightly favor 200k because a few silence-heavy samples,
especially Kiritan songs 21 and 38, regress sharply at 220k. Ten-percent
trimmed means, sample medians, active MSE, and speaker-macro deltas still favor
220k. The correct conclusion is therefore that foundation endpoint quality
clearly matured from 130k to 200k and then entered a heterogeneous plateau;
the small original panel overestimated the certainty of the 220k gain.

The immutable records are stored under `records/shadow-panel-r1/`. A separate
A-to-B lock excludes all 116 shadow-panel song units and contains 32 fixed
source/target pairs, 13 source and target speakers, and two held-out reference
songs per speaker. Its conversion metrics and blind listening remain separate
from reconstruction checkpoint ranking.

## 17. V7 sealed A-to-B conversion baseline

The conversion panel uses the immutable pair lock described above. It contains
32 source-to-target pairs from 26 source song units, with 13 source and 13
target speakers. All shadow-panel songs are excluded. Its eligible coverage is
limited to M4Singer and OpenSinger, so this is a strict paired checkpoint test,
not yet a broad cross-dataset generalization claim.

For every pair, the source ContentVec, F0, RMS, crop, inference seed, target and
source reference centroids, 32-step Heun solver, frozen ContentVec, frozen
PC-NSF, and pinned WavLM speaker verifier are identical across checkpoints.
EMA weights are evaluated under target, source, and null speaker conditions.

| step | sim(output,target) | sim(output,source) | target margin | content cosine | voicing F1 | F0 MAE (cents) |
|---:|---:|---:|---:|---:|---:|---:|
| 200k | 0.82626 | 0.81797 | 0.00829 | 0.89173 | 0.98869 | 10.43 |
| 220k | 0.82531 | 0.81564 | 0.00966 | 0.89221 | 0.98893 | 10.53 |

At 220k, changing only the condition from the source speaker to the target
speaker raises target similarity by 0.11088, lowers source similarity by
0.08549, and improves target margin by 0.19637. Target margin improves on
96.9% of pairs relative to source conditioning. Content cosine changes by only
-0.00405, while voicing and F0 remain effectively intact. The speaker path is
therefore causally active; the model has not simply ignored the target
embedding.

Absolute conversion strength is nevertheless modest. Under target conditioning
the 220k mean margin is +0.00966, the median is +0.02258, and 59.38%
of pairs have positive target margin. Source and null conditions remain strongly
source-like, consistent with the previously measured speaker information in
ContentVec.

The paired 200k-to-220k comparison shows no meaningful conversion maturation.
Target margin changes by +0.00137 (median -0.00199, 40.6% wins, 95% source-song
bootstrap CI [-0.00489, 0.00917]). Target similarity changes by -0.00096 and
source similarity by -0.00233; all key confidence intervals cross zero. The
causal target-versus-source margin effect is likewise nearly unchanged:
+0.19560 at 200k and +0.19637 at 220k.

The WavLM metric was calibrated with independent real-audio anchors. The source
query remains song-disjoint from its two-source-reference centroid. Each target
anchor is a third target-speaker song excluded from its two-reference centroid;
no query audio is included in the centroid against which it is scored.

| anchor | sim(target ref) | sim(source ref) | target margin |
|---|---:|---:|---:|
| real source | 0.71250 | 0.90153 | -0.18903 |
| real target | 0.88195 | 0.72413 | +0.15782 |

The mean anchor span is 0.34685, the median pair span is 0.13070, and 30/32
pairs have a positive span. Using `(converted margin - source margin) / (target
margin - source margin)`, the stable ratio-of-means transfer progress is 0.5689
at 200k and 0.5729 at 220k. Song-bootstrap 95% intervals are [0.4547, 0.7176]
and [0.4508, 0.7277], respectively; their paired delta is +0.0041 with interval
[-0.0140, 0.0269]. Per-pair ratios are retained but are not the primary score:
small anchor spans produce maxima above 9 even though the 200k/220k medians are
0.5812/0.6785.

During calibration, the original evaluator was found to cache centroids by
reference IDs without crop coordinates. Nineteen repeated-reference cache keys
contained different crop starts. The cache now includes ID, start frame, and
frame count. Both checkpoints were rerun; the legacy comparison remained close,
but only the crop-aware results above should be used for absolute calibration.

Together with the shadow panel, this means that reconstruction clearly improves
through 200k, but 200k-to-220k is already a plateau for both reconstruction and
measured A-to-B behavior. More steps alone have not strengthened conversion.
The authoritative records are
`records/conversion-panel-r1/step-200000-crop-aware.json`,
`step-220000-crop-aware.json`, and `speaker-calibration.json`; listening remains
required because verifier similarity is not a perceptual quality score.

## 18. Official V3 on the sealed V7 shadow panel

The published V3 1024x16 300k checkpoint and V4 220k EMA were evaluated on the
same 128 locked crops, 116 song units, and 60 physical speakers. The 256- and
512-frame inputs are prefixes of each locked 768-frame crop. Both models receive
the same numeric Gaussian-noise prefix in their native normalized spaces and use
32-step linear Euler integration. Since the published V3 checkpoint omits its
physical-speaker embedding table, the strict comparison is V3-null versus
V4-null. V3-null versus V4-correct is reported separately as operational V4
reconstruction, not as an architecture-only comparison.

Active-region raw-MSE means are:

| frames | V3 null | V4 null | V4 correct |
|---:|---:|---:|---:|
| 256 | 0.66013 | 1.48221 | 1.07957 |
| 512 | 0.59666 | 1.16992 | 0.91639 |
| 768 | 0.60606 | 1.03965 | 0.82347 |

For V4-correct minus V3-null, active-MSE mean deltas are +0.41945, +0.31973,
and +0.21741 at 256/512/768. The corresponding V4 win rates are 22.1%, 28.7%,
and 41.8%; song-bootstrap 95% intervals are [0.29214, 0.53327], [0.21082,
0.43119], and [0.12747, 0.33275]. V3 therefore remains better on active singing
at every tested context length, although the gap narrows with length. V3 active
MSE is also lower in every represented dataset at all three lengths.

Full-MSE means reverse because V3 has silence-only failures concentrated in
Kiritan:

| frames | V3 null mean / median | V4 null mean / median | V4 correct mean / median |
|---:|---:|---:|---:|
| 256 | 8.69151 / 0.56854 | 2.83347 / 1.22105 | 1.01646 / 0.74439 |
| 512 | 9.10022 / 0.52341 | 3.38234 / 0.94665 | 1.03546 / 0.67987 |
| 768 | 9.51457 / 0.55663 | 3.28914 / 0.89825 | 1.04010 / 0.62722 |

With catastrophe defined as full raw MSE above 5, V3 rates are 4.69%, 5.47%,
and 6.25%; V4-correct rates are 0%, 0.78%, and 1.56%. All six V3 failures at
256 and the seven/eight failures at 512/768 are Kiritan crops. Six are completely
silent at every length; songs 21 and 38 join at longer lengths. V3 active-region
catastrophe rate remains zero. Consequently full means demonstrate V4's stronger
silence robustness, while active metrics and paired medians are required for
singing reconstruction quality.

Active-MSE P90/P95 values are V3 0.97369/1.29148, 0.82185/1.19855, and
0.81175/1.05854 versus V4-correct 2.07118/2.73490, 1.63812/2.40943, and
1.44550/2.25348 at 256/512/768. Raw cosine means are V3 0.98951/0.98994/0.98835
versus V4-correct 0.98428/0.98698/0.98769. At 768 the paired cosine interval
crosses zero; at 256 and 512 it favors V3.

The strict V4-null path is substantially behind V3-null on active MSE: mean
deltas are +0.82208/+0.57325/+0.43359 and V4 win rates are only
2.46%/4.10%/9.84%. Correct speaker conditioning improves V4 active MSE over its
null path by -0.40263/-0.25353/-0.21618 with win rates
89.34%/85.25%/86.07%.

The authoritative output is
`records/shadow-panel-r1/v3-300k-vs-v4-220k.json`. It includes per-crop values,
mean/median/P90/P95, full/active/silence metrics, catastrophe rates,
speaker/dataset macro means, and paired dataset-song bootstrap comparisons.

## 19. Source-identity and spectral-detail diagnosis

The 512-frame Shadow-128 comparison was extended to determine whether V3's
lower active A-to-A error merely rewards source-speaker leakage. The identity
lock uses two different-song source references and four unrelated speaker
centroids per crop. The first unrelated speaker is also used as V4's wrong
condition. Of 128 wrong speakers, 120 are from the same dataset; Kiritan's eight
crops require a cross-dataset fallback because Kiritan contains only one
physical speaker. The integration noise, crop positions, solver, PC-NSF, and
pinned WavLM verifier are otherwise fixed. Six all-silence crops remain in the
speaker audit but are excluded from active-MSE and DCT statistics.

### 19.1 Speaker similarity does not explain V3's precision

Across all 128 crops, WavLM speaker metrics are:

| model | source similarity | unrelated similarity | source margin |
|---|---:|---:|---:|
| V3 null | 0.82773 | 0.74585 | 0.08188 |
| V4 correct | 0.84796 | 0.76556 | 0.08240 |

V4-correct is +0.02023 more similar to the source, but also +0.01972 more
similar to unrelated speakers. Its discriminative source-margin gain is only
+0.00052, with song-bootstrap 95% CI [-0.00619, +0.00699]. The two models are
therefore indistinguishable on this verifier's source-versus-impostor margin.

For the 122 active crops, define V3 MSE advantage as
`MSE(V4-correct) - MSE(V3-null)` and V3 speaker advantage as
`similarity(V3, source) - similarity(V4, source)`. Their Pearson correlation is
0.20269, but rank correlation is only 0.02400 with song-bootstrap 95% CI
[-0.16415, 0.20646]. Replacing raw source similarity with source margin removes
even the weak linear relationship: Pearson -0.01948, Spearman -0.01574, CI
[-0.19360, 0.16290]. Excluding Kiritan leaves the raw-similarity Spearman at
0.03145. The data do not support source-identity similarity as the main cause
of V3's active-MSE advantage.

### 19.2 The V4 speaker condition is causal

Changing only V4's condition from the correct source speaker to a wrong speaker
produces the following paired deltas:

| wrong minus correct | mean | median | song-bootstrap mean 95% CI |
|---|---:|---:|---:|
| active raw MSE | +0.48427 | +0.43265 | [+0.40103, +0.57159] |
| source similarity | -0.02821 | -0.01088 | [-0.04135, -0.01577] |
| wrong-speaker similarity | +0.04937 | +0.02720 | [+0.03450, +0.06535] |
| wrong-minus-source margin | +0.07757 | +0.04000 | [+0.05526, +0.10160] |

Wrong conditioning increases wrong-speaker similarity in 74.22% of crops and
the wrong-minus-source margin in 81.25%; it beats correct conditioning on
active MSE in only 7.38%. V4 is not ignoring the explicit speaker path. It
causally trades A-to-A reconstruction for movement toward the requested wrong
identity.

### 19.3 V3 wins voiced fine spectral structure

The active error separates sharply by voicing:

| region | V3 null MSE | V4 correct MSE | V4 minus V3 |
|---|---:|---:|---:|
| voiced | 0.60172 | 0.99090 | +0.38918 |
| unvoiced active | 0.53043 | 0.46829 | -0.06214 |

V3's total active advantage is almost perfectly aligned with its voiced-frame
advantage: Pearson 0.99514, Spearman 0.99488, song-bootstrap Spearman CI
[0.98901, 0.99624]. V4 is actually better in unvoiced active regions.

An orthonormal DCT-II along the 128-bin log-mel frequency axis further splits
smooth spectral-envelope coefficients from finer frequency structure. Each
low/high pair sums exactly to active MSE:

| cutoff | component | V3 null | V4 correct | V4 minus V3 | mean 95% CI |
|---:|---|---:|---:|---:|---:|
| 8 | low | 0.23548 | 0.19424 | -0.04124 | [-0.08244, -0.00969] |
| 8 | high | 0.35955 | 0.72617 | +0.36662 | [+0.27195, +0.47118] |
| 16 | low | 0.32028 | 0.27761 | -0.04267 | [-0.08767, -0.00610] |
| 16 | high | 0.27475 | 0.64280 | +0.36805 | [+0.27687, +0.47305] |
| 32 | low | 0.42934 | 0.45771 | +0.02837 | [-0.02973, +0.08076] |
| 32 | high | 0.16569 | 0.46270 | +0.29701 | [+0.22815, +0.37201] |

At cutoffs 8 and 16, V4 is significantly better on the smoothest component,
while V3 wins the high component. At cutoff 32, 0.29701 of the total 0.32538
MSE gap, about 91%, remains in the high component; V3 wins this component on
all 122 active crops. Total advantage and DCT-16 high-component advantage have
Pearson 0.92354 and Spearman 0.94880, whereas DCT-16 high advantage versus
source-margin advantage has Spearman -0.03065 with CI [-0.20629, 0.14219].

Per-bin comparison gives the same localization: V4 wins 36/128 bins, including
bin 0 and bins 93-127, while V3's largest gains lie around bins 27-55,
especially 35-53. This is evidence for a voiced, mid-band, fine-frequency
precision gap. DCT does not by itself prove that the error is harmonic, but its
voiced localization makes harmonic reconstruction the leading hypothesis.

Representative crops do not show a monotonic identity explanation. V4 strongly
wins `M4Singer:红玫瑰` (-2.0745 active-MSE delta), while V3 strongly wins
`M4Singer:渔光曲` (+2.6771), `M4Singer:白发亲娘` (+2.2890), and
`GTSinger:Frühlingsglaube` (+1.9556). Their source-similarity and source-margin
advantages have mixed signs, while their voiced and high-DCT errors track the
overall result.

The current evidence therefore rejects the narrow claim that V3 wins A-to-A
because it simply preserves more source timbre. V4 has real, effective speaker
control and a better smooth-envelope/unvoiced reconstruction result, but V3 is
substantially more precise on voiced fine spectral structure. Follow-up work
should examine raw velocity error by voiced frequency region, normalization
weighting, and RF trajectory behavior before changing speaker conditioning or
introducing a content adversary. This audit alone is not a reason to alter the
running training recipe.

The authoritative outputs are
`records/shadow-panel-r1/identity-512.lock.json` and
`records/shadow-panel-r1/identity-diagnostic-512.json`.

## 20. EMA, solver, and local-velocity spectral diagnosis

The same locked 512-frame Shadow-128 inputs were used to test three possible
explanations for V4's voiced fine-spectrum deficit. DCT bands are fixed at
0-15, 16-31, and 32-127. All endpoint numbers below are per-sample means over
voiced frames and use correct speaker conditioning for V4.

### 20.1 EMA is helping, not hiding late detail

At 220k, raw and EMA weights under identical Euler-32 integration give:

| state | DCT 0-15 | DCT 16-31 | DCT 32-127 | high-band correlation |
|---|---:|---:|---:|---:|
| 220k raw | 0.30389 | 0.26751 | 0.62747 | 0.40121 |
| 220k EMA | 0.28172 | 0.19492 | 0.50850 | 0.53086 |

Raw-minus-EMA high-band MSE is +0.11897 with song-bootstrap 95% CI
[+0.05020, +0.18779]. Raw is significantly worse in all three bands. Its
high-band generated/target RMS ratio is 0.9562 versus EMA's 0.9707, but the
larger difference is correlation. EMA improves structure rather than merely
restoring energy.

The EMA-lag control is also negative. From 220k EMA to 230k EMA, the three band
deltas are -0.00062, -0.00508, and -0.00186; every bootstrap interval crosses
zero. A later EMA does not converge toward the worse 220k raw result. The
fine-detail deficit is therefore not explained by `ema_decay=0.9999` smoothing
away a newly matured raw model.

### 20.2 More accurate integration does not restore detail

The 220k EMA solver sweep is:

| solver | approximate NFE | DCT 0-15 | DCT 16-31 | DCT 32-127 |
|---|---:|---:|---:|---:|
| Euler 16 | 16 | 0.27264 | 0.18711 | **0.48280** |
| Euler 32 | 32 | 0.28172 | 0.19492 | 0.50850 |
| Heun 16 | 31 | 0.28900 | 0.19856 | 0.51596 |
| Euler 64 | 64 | 0.28711 | 0.19928 | 0.52363 |
| Heun 32 | 63 | 0.29183 | 0.20156 | 0.52977 |

Relative to Euler 32, Euler 16 improves the high band by -0.02571, CI
[-0.03619, -0.01605]. Euler 64 worsens it by +0.01513, and Heun 32 by
+0.02127; both intervals exclude zero. At nearly equal cost, Heun 16 is worse
than Euler 32 and Heun 32 is worse than Euler 64. This rules against ordinary
32-step Euler truncation error as the missing-detail mechanism. Coarser Euler
appears to add a favorable numerical bias to an imperfect learned field;
following that field more accurately moves the endpoint farther from the paired
target.

### 20.3 The deficit already exists in the local vector field

Local velocity was evaluated on each model's native RF interpolation using the
same numerical Gaussian noise, then transformed into raw-log-mel velocity units
before the DCT. The paired target paths are native to each parameterization, so
this comparison measures the complete model-plus-normalization recipe rather
than architecture alone.

Voiced DCT-32-127 velocity NMSE is:

| t | V3 null | V4 220k EMA correct | V4 minus V3 |
|---:|---:|---:|---:|
| .10 | 0.00365 | 0.03152 | +0.02787 |
| .25 | 0.00504 | 0.03968 | +0.03464 |
| .50 | 0.01035 | 0.06756 | +0.05721 |
| .75 | 0.03376 | 0.16122 | +0.12746 |
| .90 | 0.13339 | 0.39177 | +0.25838 |
| .95 | 0.30112 | 0.58434 | +0.28322 |

Every paired song-bootstrap interval is strictly positive. For example, the
V4-minus-V3 intervals are [+0.02697, +0.02883] at t=.10, [+0.12449,
+0.13041] at t=.75, and [+0.27652, +0.28997] at t=.95. The 16-31 band is also
worse for V4 at every t. Conversely, V4 becomes better in DCT 0-15 at high t:
its low-band NMSE is 0.18803 versus V3's 0.22102 at t=.90, and 0.22946 versus
0.35973 at t=.95. This reproduces the endpoint split inside the local vector
field: V4 is strong on smooth structure but weak on fine modes.

At endpoint, high-band generated/target RMS is not strongly attenuated: V3 is
0.9648 and V4 is 0.9707. The decisive difference is coefficient correlation,
0.8257 versus 0.5309. V4 generates approximately the right total fine-spectrum
energy but places or signs much of it incorrectly. At high t the local field
develops both errors: at t=.95, V4's high-band RMS ratio/correlation are
0.6325/0.6434 versus V3's 0.8351/0.8359. A simple high-frequency gain is
therefore not an adequate fix.

### 20.4 Pitch stratification

The endpoint high-band comparison was independently rerun and stratified by
voiced-frame F0. The locked panel's F0 quartile boundaries are 221.46, 287.07,
and 372.60 Hz:

| F0 quartile | V4-minus-V3 high-band MSE | mean 95% CI | correlation delta |
|---|---:|---:|---:|
| Q1 | +0.14687 | [+0.08827, +0.21583] | -0.11842 |
| Q2 | +0.24271 | [+0.16945, +0.32610] | -0.23216 |
| Q3 | +0.32388 | [+0.23838, +0.42048] | -0.33518 |
| Q4 | +0.42263 | [+0.29652, +0.55703] | -0.34566 |

The gap grows monotonically with F0. It is not concentrated in fast pitch
transitions: using absolute pitch movement per 11.61 ms frame, high-band gaps
are +0.39491 for stable `<0.1` semitone frames, +0.25095 for 0.1-0.5, and
+0.08568 for `>=0.5`. All intervals exclude zero. The current evidence points
to sustained pitched fine-structure precision, especially at high F0, rather
than a narrow transient or vibrato-tracking failure.

Stratifying by target DCT-32-127 RMS gives gaps of +0.06672, +0.24508,
+0.37332, and +0.49814 from the lowest to highest target-detail quartile. This
is a spectral-detail-energy proxy, not a direct HNR measurement, but it confirms
that V3 benefits much more when the target contains strong fine structure.

Together, these controls move the primary cause upstream of sampling: the V4
recipe has learned a less accurate voiced mid/high-DCT vector field, and the
problem becomes acute near the data endpoint. Next experiments should isolate
raw-space frequency weighting under per-channel normalization and inspect
F0-conditioned harmonic alignment. Solver or EMA changes cannot close the
observed gap, and an amplitude-only auxiliary loss is not yet justified.

The authoritative outputs are
`records/shadow-panel-r1/spectral-detail-audit-512.json` and
`records/shadow-panel-r1/spectral-pitch-audit-512.json`.

## 21. Training-crop ceiling and output-head diagnosis

Two final controls distinguish song-disjoint generalization from an in-sample
optimization/representation limit. Main training was deliberately stopped at
logged step 265,920 after the complete 265k resume checkpoint had been written;
all checkpoints and records were retained.

### 21.1 Matched training crops reproduce the local-field gap

A deterministic training lock contains 128 crops from 126 training songs and
60 physical speakers. Its per-dataset and per-speaker crop counts exactly match
Shadow-128: 68 GTSinger, 25 M4Singer, 27 OpenSinger, and 8 Kiritan crops. Each
speaker contributes different songs before secondary segments are selected.
The panel is therefore not a small hand-picked easy-training set.

Voiced DCT-32-127 local velocity NMSE is:

| t | train V3 | train V4 | train gap | Shadow gap |
|---:|---:|---:|---:|---:|
| .10 | 0.00365 | 0.02435 | +0.02071 | +0.02787 |
| .25 | 0.00508 | 0.03057 | +0.02548 | +0.03464 |
| .50 | 0.01054 | 0.05756 | +0.04702 | +0.05721 |
| .75 | 0.03460 | 0.15545 | +0.12085 | +0.12746 |
| .90 | 0.13675 | 0.39710 | +0.26035 | +0.25838 |
| .95 | 0.30717 | 0.59457 | +0.28740 | +0.28322 |

Every training-crop V4-minus-V3 song-bootstrap interval excludes zero. At
low/mid t, training reduces but does not close the gap. At t=.90 and .95 the
training and Shadow gaps are effectively the same. For example, train t=.90 CI
is [+0.25467, +0.26603], while Shadow is [+0.25213, +0.26464]. V4 therefore
fails to fit these fine spectral velocity modes even on its own training songs.
The primary problem is not song-disjoint generalization; it is an
optimization, objective-geometry, parameterization, or representation ceiling.

### 21.2 Output head is mildly tilted, but Adam variance is not suppressing it

The final output matrices were converted to raw-log-mel output units and
projected along their 128 mel rows with the same orthonormal DCT. Mean row RMS,
relative to DCT 0-15, is:

| model/state | DCT 0-15 | DCT 16-31 | DCT 32-127 |
|---|---:|---:|---:|
| V3 300k | 1.000 | 0.887 | 0.753 |
| V4 220k raw | 1.000 | 0.911 | 0.662 |
| V4 265k raw | 1.000 | 0.914 | 0.660 |

V4's high-DCT output directions are modestly smaller relative to its low band,
but they are not frozen or collapsing. From 220k to 265k, absolute row RMS
grows 4.89%, 5.23%, and 4.49% in the low, mid, and high bands respectively;
the relative spectral profile remains constant.

The full V4 checkpoints contain Adam state. Under a diagonal
parameter-coordinate covariance approximation, projected `sqrt(exp_avg_sq)`
is flat across DCT bands:

| step | DCT 0-15 | DCT 16-31 | DCT 32-127 |
|---:|---:|---:|---:|
| 220k | 9.1245e-5 | 9.1349e-5 | 9.1327e-5 |
| 265k | 7.3962e-5 | 7.4086e-5 | 7.4048e-5 |

There is no high-band inflation in Adam's denominator. The projected
preconditioned first-moment RMS is lower in the high band, at 0.512x the low
band at 220k and 0.518x at 265k. Fixed-panel instantaneous gradients show the
same direction: V4 high/low gradient RMS is 0.444 at 220k and 0.485 at 265k.
Thus the optimizer receives a weaker coherent high-DCT update, rather than
selectively suppressing an equally strong signal through a larger second
moment.

The official V3 checkpoint has no optimizer state, so a V3 Adam-moment
comparison is impossible. Its current fixed-panel high/low output-head gradient
ratio is only 0.028 despite superior high-DCT reconstruction, demonstrating
that a small late gradient is not itself evidence of poor learning; it may mean
the mode is already fitted. Likewise, absolute output-head norm cannot isolate
the head from upstream hidden-state scaling. The head audit therefore rules out
a simple Adam second-moment failure but does not prove that the backbone alone
is responsible.

Combined with the training-crop result, the highest-value controlled ablation
is now raw-space frequency loss weighting under the V4 per-channel normalized
input/output representation. Giving only the output head a larger LR is poorly
motivated: all bands are growing together, and the missing high-band signal is
already visible in local velocity targets and gradients. A second useful
ablation is the V3/V4 parameterization change at fixed normalization, but that
requires a new clean training branch rather than another checkpoint audit.

The authoritative outputs are
`records/shadow-panel-r1/train-panel-matched.lock.json`,
`records/shadow-panel-r1/train-vs-shadow-local-spectral-512.json`, and
`records/shadow-panel-r1/output-head-audit.json`.
