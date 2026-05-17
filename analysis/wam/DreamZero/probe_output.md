================================================================================
  DreamZero 探针代码测试
  World-Action Model — Wan2.2-TI2V-5B
================================================================================
  Device: cuda
  PyTorch: 2.9.1+cu128

================================================================================
  模块1: FlowMatchScheduler (Shifted Flow Matching)
================================================================================
  [FlowMatchScheduler]
    输入  original_samples: (1, 16, 4, 60, 104) dtype=torch.float32
    输入  noise: (1, 16, 4, 60, 104) dtype=torch.float32
    输入  timestep: (1,) dtype=torch.float32
    输出  noisy_samples: (1, 16, 4, 60, 104) dtype=torch.float32
    输出  v_target: (1, 16, 4, 60, 104) dtype=torch.float32
    输出  denoised_step: (1, 16, 4, 60, 104) dtype=torch.float32
    参数  总计: 0, 可训练: 0
    备注  推理sigma范围: [0.2500, 1.0000], 推理步数: 16

================================================================================
  模块2: T5 Text Encoder (冻结)
================================================================================
  [T5TextEncoder]
    输入  input_ids: (1, 32) dtype=torch.int64
    输出  text_emb: (1, 32, 4096) dtype=torch.float32
    参数  总计: 1.45B, 可训练: 1.45B
    备注  实际24层, 探针2层; 实际参数~4,700M

================================================================================
  模块3: CLIP Image Encoder (冻结)
================================================================================
  [CLIPImageEncoder]
    输入  image: (1, 3, 336, 336) dtype=torch.float32
    输出  clip_context: (1, 1, 1280) dtype=torch.float32
    参数  总计: 40.85M, 可训练: 40.85M
    备注  实际32层ViT-H/14, 探针2层; 实际参数~355M

================================================================================
  模块4: Wan Video VAE (CausalConv3d, 冻结)
================================================================================
  [WanVideoVAE]
    输入  video: (1, 3, 4, 480, 832) dtype=torch.float32
    输出  latent: (1, 16, 1, 30, 52) dtype=torch.float32
    输出  reconstruction: (1, 3, 4, 480, 832) dtype=torch.float32
    参数  总计: 1.19M, 可训练: 1.19M
    备注  压缩: (1, 3, 4, 480, 832) → (1, 16, 1, 30, 52); 实际参数~269M

================================================================================
  模块5: Multi-Embodiment Action Encoder (核心创新)
================================================================================
  [MultiEmbodimentActionEncoder]
    输入  actions: (1, 32, 14) dtype=torch.float32
    输入  timesteps: (1,) dtype=torch.float32
    输入  cat_ids: (1,) dtype=torch.int64
    输出  action_emb: (1, 32, 64) dtype=torch.float32
    参数  总计: 428.03K, 可训练: 428.03K
    备注  CategorySpecificLinear: 每个具身类别独立W/b

================================================================================
  模块6: Causal Chunk Self-Attention (核心创新)
================================================================================
  [CausalWanSelfAttention]
    输入  x (clean+noisy+action+state): (1, 1793, 5120) dtype=torch.float32
    输出  attn_out: (1, 1793, 5120) dtype=torch.float32
    参数  总计: 104.89M, 可训练: 104.89M
    备注  Teacher Forcing: clean因果, noisy/action看clean+自己, state只看自己

================================================================================
  模块7: Wan DiT Block
================================================================================
  [WanDiTBlock]
    输入  x: (1, 1793, 5120) dtype=torch.float32
    输入  context: (1, 32, 5120) dtype=torch.float32
    输出  x_out: (1, 1793, 5120) dtype=torch.float32
    参数  总计: 422.17M, 可训练: 422.17M
    备注  Self-Attn + Cross-Attn + SwiGLU FFN; 实际参数~262M/层

================================================================================
  模块8: CausalWanModel (完整DiT)
================================================================================
  [CausalWanModel]
    输入  x: (1, 1793, 36) dtype=torch.float32
    输入  context: (1, 32, 5120) dtype=torch.float32
    输入  timestep: (1,) dtype=torch.float32
    输入  cat_ids: (1,) dtype=torch.int64
    输出  v_pred: (1, 1793, 16) dtype=torch.float32
    参数  总计: 1.29B, 可训练: 1.29B
    备注  探针3层, 实际40层; 实际参数估算~16.91B

================================================================================
  模块9: Camera Controller (SimpleAdapter)
================================================================================
  [SimpleAdapter]
    输入  plucker: (1, 2, 6, 384, 672) dtype=torch.float32
    输出  camera_emb: (1, 1024, 2, 23, 41) dtype=torch.float32
    参数  总计: 41.29M, 可训练: 41.29M
    备注  Plucker坐标 → PixelUnshuffle → Conv → ResBlocks

================================================================================
  模块10: WANPolicyHead (完整动作头)
================================================================================
  [WANPolicyHead]
    输入  x_latent: (1, 1793, 36) dtype=torch.float32
    输入  context: (1, 32, 5120) dtype=torch.float32
    输入  timestep: (1,) dtype=torch.float32
    输入  cat_ids: (1,) dtype=torch.int64
    输出  v_pred: (1, 1793, 16) dtype=torch.float32
    参数  总计: 2.83B, 可训练: 2.83B
    备注  探针3层DiT; 实际40层+T5+CLIP+VAE ≈ 10,300M

================================================================================
  DreamZero 参数汇总表
================================================================================
  模块                                       参数量          状态      
  ---------------------------------------- ------------ --------
  T5-XXL Text Encoder                      ~4,700M      冻结      
  CLIP ViT-H/14 Image Encoder              ~355M        冻结      
  WanVideoVAE (CausalConv3d)               ~269M        冻结      
  CausalWanModel (40层DiT)                  ~5,000M      LoRA    
    ├─ Self-Attn (40层)                     ~4,200M      LoRA    
    ├─ Cross-Attn (40层)                    ~400M        LoRA    
    ├─ FFN (40层)                           ~400M        LoRA    
    └─ Embed + Norm + Proj                 ~2M          LoRA    
  Multi-Embodiment Action Enc              ~2,100K      可训练     
  Action Decoder                           ~82K         可训练     
  Camera Controller                        ~8,400K      可训练     
  LoRA (rank=4, DiT上)                      ~20,000K     可训练     
  ---------------------------------------- ------------ --------
  总计                                       ~10,300M             
  可训练参数                                    ~30,000K             
  冻结参数                                     ~10,270M             

================================================================================
  DreamZero 数据流图
================================================================================

  训练阶段 (Teacher Forcing):
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  text_ids ──→ [T5 Encoder] ──→ context (B,L_text,4096)            │
  │                                   │                                 │
  │  video ──→ [VAE Encoder] ──→ clean_latent (B,16,T/4,H/8,W/8)     │
  │                                   │                                 │
  │                                   ├──→ x_0 (clean)                 │
  │                                   │      │                         │
  │  noise ──────────────────────────────────┤                         │
  │                                   │      ↓                         │
  │  timestep ──→ [Scheduler] ──→ sigma     x_t = (1-σ)x_0 + σ·noise │
  │                                   │      │                         │
  │  actions ──→ [Action Enc] ──→ act_emb   │                         │
  │                                   │      │                         │
  │  state ──→ state_tokens ──→ state_emb   │                         │
  │                                   │      │                         │
  │                                   ↓      ↓                         │
  │                          [concat: clean | noisy | action | state]  │
  │                                   │                                 │
  │                                   ↓                                 │
  │                          [CausalWanModel × 40层]                   │
  │                           ├─ Causal Chunk Self-Attn                │
  │                           ├─ Cross-Attn (← context)                │
  │                           └─ SwiGLU FFN                            │
  │                                   │                                 │
  │                                   ↓                                 │
  │                          v_pred (速度场)                            │
  │                                   │                                 │
  │                          loss = MSE(v_pred, noise - x_0)           │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  推理阶段:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  x_T = noise (纯噪声)                                              │
  │         │                                                           │
  │         ↓  for t = T, T-1, ..., 1:                                 │
  │  [CausalWanModel] ← context, timestep                              │
  │         │                                                           │
  │         ↓                                                           │
  │  v_pred = model(x_t, context, t)                                   │
  │         │                                                           │
  │         ↓                                                           │
  │  x_{t-1} = x_t + v_pred · (σ_{t-1} - σ_t)  (Euler步)             │
  │         │                                                           │
  │         └──→ 重复16步                                               │
  │                                                                     │
  │  x_0 ──→ [Action Decoder] ──→ action_pred (B, T, action_dim)      │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  Causal Chunk Attention 可见性:
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Clean:    [B0] [B1] [B2] [B3]                                 │
  │             ↓    ↓    ↓    ↓                                    │
  │  B0看:    [✓]                                                  │
  │  B1看:    [✓]  [✓]                                             │
  │  B2看:    [✓]  [✓]  [✓]                                        │
  │  B3看:    [✓]  [✓]  [✓]  [✓]   ← 因果: 只看过去块             │
  │                                                                  │
  │  Noisy_i: 看 clean_0..i + noisy_i + action_i + state_i         │
  │  Action_i: 看 clean_0..i + noisy_i + action_i + state_i        │
  │  State_i: 只看自己 (条件信息, 不参与注意力)                      │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
  

================================================================================
  核心创新总结
================================================================================
  [模块6] Causal Chunk Attention
         图像块因果注意力 + 动作/状态与对应图像块交叉注意力, teacher forcing训练避免误差累积

  [模块5] Multi-Embodiment Action Encoder
         CategorySpecificLinear, 每个具身类别独立权重, 多机器人共享模型

  [模块1+10] 视频-动作联合 Flow Matching
         统一世界模型+动作模型, 视频/动作共享去噪过程, decoupled noise可独立调度

  [模块4] CausalConv3d VAE
         因果3D卷积, 时间维度只看过去帧, 空间8x/时间4x压缩

  [模块1] Shifted Flow Matching
         shift参数控制sigma分布, 低噪声区间更密集, 提升生成质量

  [模块10] Decoupled Noise
         视频/动作可使用不同噪声调度, 推理时视频保持噪声而动作完全去噪

================================================================================
  所有模块测试通过!
================================================================================
