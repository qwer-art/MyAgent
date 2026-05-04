# Open-VINS 框架理解与工具开发

## 坐标系命名规范

1. **旋转矩阵**: `R_XtoY` — 从坐标系 X 到坐标系 Y 的旋转
2. **四元数**: `q_XtoY` — 从坐标系 X 到坐标系 Y 的 JPL 四元数
3. **位置向量**: `p_XinY` — 坐标系 X 原点在坐标系 Y 中的坐标
4. **相机-IMU外参**: `T_imu_cam` — 旋转 `R_CtoI`，平移 `p_CinI`

### 坐标系定义

| 符号 | 名称 | 说明 |
|------|------|------|offset_R_L_I ↔ R_LtoI (状态中的外参旋转)
offset_T_L_I ↔ p_LinI (状态中的外参平移)
| I | IMU | IMU本体坐标系 |
| L | LiDAR | 激光雷达坐标系 |
| C | Camera | 相机坐标系 |
| W | World | 世界坐标系（固定坐标系） |
| G | Global | 全局坐标系（与W等价，用于VIO） |
| B | Base | 机器人基座坐标系（用于重力对齐） |

### 代码变量名与数学符号对应关系

| 代码变量名 | 数学符号 | 说明 |
|-----------|---------|------|
| `Lidar_R_wrt_IMU` | `R_LtoI` | LiDAR到IMU的旋转 |
| `Lidar_T_wrt_IMU` | `p_LinI` | LiDAR原点在IMU坐标系中的位置 |
| `offset_R_L_I` | `R_LtoI` | 状态中的外参旋转 |
| `offset_T_L_I` | `p_LinI` | 状态中的外参平移 |

---

## Assumptions (需确认)

- 用户熟悉ROS开发环境
- 目标是理解框架结构并开发配套工具链
- 工作环境已有ROS和基础依赖
- 最终目的是能够录制、解析、可视化SLAM数据

**不确定项：**
- 是否需要支持多传感器配置？还是只针对特定配置？
- 录制数据的目标场景是什么？（室内/室外、时长、数据量）

---

## 整体流程

```
步骤1: 代码结构分析
  → verify: topic列表完整、结构图符合实际

步骤2: 开发init_env.sh
  → verify: 脚本执行成功、编译通过

步骤3: 开发record_data.sh
  → verify: 录制数据包含所有必需topic

步骤4: 开发parser_bag.py
  → verify: 解析数据与原始bag一致

步骤5: 开发pangolin_viz.py
  → verify: 可视化效果正确
```

---

## 步骤1: 整理代码整体结构图

**Goal:** 输出框架入口、通讯结构、topic/msg含义

**Success criteria:**
- 所有入口函数已识别
- 所有topic已列出并标注类型
- 发布/订阅关系已绘制

### 执行计划

```
1. 搜索入口函数
   → verify: grep "int main" 结果与文档一致

2. 搜索topic定义
   → verify: grep "advertise|subscribe" 结果覆盖文档列表

3. 提取msg字段含义
   → verify: 抽查3个msg的字段与源码定义一致

4. 绘制通讯结构图
   → verify: 用户确认逻辑流向合理
```

### 校验点1: 结构图与topic列表

**自动校验：**
```bash
# 1. 检查入口函数完整性
grep -rn "int main" --include="*.cpp" --include="*.cc" > /tmp/entries.txt
# 对比文档中列出的入口，标记遗漏

# 2. 检查topic完整性
grep -rn "advertise\|subscribe" --include="*.cpp" --include="*.py" > /tmp/topics.txt
# 提取topic名称，与文档列表做diff

# 3. 检查launch文件中的topic
grep -rn "topic" --include="*.launch" >> /tmp/topics.txt
# 补充遗漏的topic
```

**需人工校验：**
- [ ] 通讯结构图的算法逻辑流向是否合理
- [ ] msg字段的具体物理含义（如坐标系、单位等）
- [ ] 配置文件中的参数含义

**校验结果记录：**
```
执行时间: _______
自动校验: PASS / FAIL
遗漏项: _______
人工确认: _______
```

---

## 步骤2: 开发init_env.sh

**Goal:** 环境初始化、编译、别名设置

**Success criteria:**
- 脚本无语法错误
- 依赖安装完整
- 编译成功
- 别名可用

### 执行计划

```
1. 提取依赖列表
   → verify: package.xml中的依赖已包含在脚本中

2. 编写安装和编译命令
   → verify: bash -n init_env.sh 通过

3. 设置别名
   → verify: 别名符合常用习惯

4. 测试执行
   → verify: 在干净环境执行成功（返回码0）
```

### 校验点2: 环境脚本

**自动校验：**
```bash
# 1. 语法检查
bash -n init_env.sh

# 2. 依赖完整性
# 解析package.xml提取依赖
grep "<depend>" package.xml | sed 's/.*<depend>\(.*\)<\/depend>.*/\1/' > /tmp/deps.txt
# 对比脚本中的apt install列表

# 3. 编译测试
./init_env.sh
echo $?  # 应为0
```

**需人工校验：**
- [ ] 编译失败时的错误排查
- [ ] 环境变量路径是否正确
- [ ] 别名命名是否合理

**校验结果记录：**
```
执行时间: _______
语法检查: PASS / FAIL
编译结果: PASS / FAIL
错误信息: _______
```

---

## 步骤3: 开发record_data.sh

**Goal:** 录制算法所需的所有topic数据

**Success criteria:**
- 覆盖所有算法订阅的topic
- 录制命令正确
- bag文件可读

### 执行计划

```
1. 提取算法订阅的topic列表
   → verify: 与步骤1识别结果一致

2. 编写录制命令
   → verify: bash -n record_data.sh 通过

3. 测试录制
   → verify: rosbag info 显示所有topic都有数据
```

### 校验点3: 录制脚本

**自动校验：**
```bash
# 1. 语法检查
bash -n record_data.sh

# 2. topic覆盖度
# 从步骤1提取订阅topic列表
# 对比录制脚本中的topic列表
# 计算覆盖率 = 已覆盖 / 总数

# 3. 实际录制测试（需传感器）
./record_data.sh --duration 5
rosbag info recorded.bag
# 检查每个topic的messages数量 > 0
```

**需人工校验：**
- [ ] 无传感器时无法测试，需在有设备环境验证
- [ ] topic优先级（哪些是必须的）
- [ ] 录制参数（频率、压缩等）

**校验结果记录：**
```
执行时间: _______
语法检查: PASS / FAIL
topic覆盖率: ____%
录制测试: PASS / FAIL / SKIP(无设备)
遗漏topic: _______
```

---

## 步骤4: 开发parser_bag.py

**Goal:** 解析bag文件，提取数据为可读格式

**Success criteria:**
- 所有字段已解析
- 数据类型正确
- 无数据丢失

### 执行计划

```
1. 解析bag文件结构
   → verify: 所有topic都能读取

2. 提取msg字段
   → verify: 字段名与msg定义一致

3. 保存为可读格式
   → verify: 输出文件格式正确

4. 数据完整性检查
   → verify: 消息数与rosbag info一致
```

### 校验点4: 解析结果

**自动校验：**
```bash
# 1. 字段完整性
# 解析bag的msg定义
python3 -c "import rosbag; bag = rosbag.Bag('test.bag'); ..."
# 对比解析脚本输出的字段

# 2. 数据类型验证
# 抽取前10条消息，检查类型

# 3. 数据量一致性
rosbag info test.bag | grep "messages:"
# 对比解析出的消息数

# 4. 时间戳连续性
# 检查时间戳单调递增
# 标记异常跳变（间隔 > 10倍中位数）

# 5. 数据范围合理性
# 四元数模接近1
# 无NaN/Inf
```

**需人工校验：**
- [ ] 抽查3-5条数据，人工对比原始bag和解析结果
- [ ] 时间戳跳变的原因判断
- [ ] 坐标系是否正确

**校验结果记录：**
```
执行时间: _______
字段完整性: PASS / FAIL
数据量一致性: PASS / FAIL
时间戳异常: ____处
人工抽查: PASS / FAIL
问题字段: _______
```

---

## 步骤5: 开发pangolin_viz.py

**Goal:** 可视化解析后的数据

**Success criteria:**
- 数据正确加载
- 可视化窗口正常显示
- 交互流畅

### 执行计划

```
1. 加载解析数据
   → verify: 数据维度正确

2. 创建可视化窗口
   → verify: 窗口正常创建

3. 渲染轨迹和点云
   → verify: 无渲染错误

4. 测试交互
   → verify: 鼠标旋转/缩放正常
```

### 校验点5: 可视化效果

**自动校验：**
```bash
# 1. 依赖检查
python3 -c "import pangolin"

# 2. 数据加载
python3 pangolin_viz.py --test-load
# 检查数据维度

# 3. 坐标系验证
# 检查轨迹起点在原点附近
# 检查尺度在合理范围

# 4. 渲染测试
python3 pangolin_viz.py --test-render
# 检查是否有错误输出
```

**需人工校验：**
- [ ] 轨迹是否平滑合理
- [ ] 点云是否稠密合理
- [ ] 颜色/视角设置
- [ ] 交互流畅度

**校验结果记录：**
```
执行时间: _______
依赖检查: PASS / FAIL
数据加载: PASS / FAIL
渲染测试: PASS / FAIL
视觉效果: _______
```

---

## 校验原则

参考 CLAUDE.md 的要求：

1. **不假设** - 不确定的地方标记出来，寻求确认
2. **最小化** - 只做必要的校验，不过度设计
3. **可验证** - 每个步骤都有明确的成功标准
4. **循环改进** - 校验失败时修复后重新验证

**校验优先级：**
- 校验点1、3、4 最关键（影响后续步骤的正确性）
- 校验点2、5 相对次要（可通过执行测试验证）

**自动校验 vs 人工校验：**
- 能用命令/脚本验证的，优先自动校验
- 涉及语义理解、主观判断的，才需人工校验
- 自动校验失败时，提供详细信息辅助人工判断
