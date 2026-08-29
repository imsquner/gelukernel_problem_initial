# GeluKernel 算子深度实现文档

> 华为昇腾（Ascend）CANN 9.2 上的 GELU 激活函数算子，Ascend C Direct Invocation 模式。
> 参赛成绩：**CANNJudge 算子挑战赛 S2 · GeluKernel，84.68 分 / 第 1 名**（通过率 100%）。

---

## 一、项目概述

### 1.1 算子功能

对输入张量每个元素应用 GELU 激活：


<img src="gelu.png" alt="Gelu" width="100%" />

### 1.2 精度要求（赛题规定）

| 数据类型 | 相对误差 | 绝对误差 |
|---|---|---|
| float32 | < 1e-4 | < 1e-4 |
| float16 | < 1e-3 | < 1e-3 |

### 1.3 约束

- 仅支持 **float16 / float32**；任意多维；元素总数 $N \in [1, 10240]$（含批次维度）；支持非 32 字节对齐。

### 1.4 工程文件

```
├── CMakeLists.txt      # 构建配置（ASC 语言，--npu-arch=dav-2201）
├── kernel.asc          # ★ 算子实现（本次全部改动都在此文件）
├── main.asc            # host 侧本地测试（aclnn 加载/运行）
├── data_utils.h        # 文件读写工具
├── run.sh              # 编译 + 生成数据 + 运行 + 验证
└── scripts/            # 数据生成 / 参考实现 / 结果验证
```

---

## 二、Ascend C 经典算子结构速览

### 2.1 两级分工

```
main.asc (host 侧)                kernel.asc (device 侧)
─────────────────────             ─────────────────────────
aclInit / aclrtMalloc       ──▶   run_kernel(...)   ← 真正的"入口"
读入 input_x.bin                      │
    │                                 ├─ 解析 shape / dtype / 核数
    ▼                                 ├─ 计算分块 (blockNum / perBlock)
run_kernel(input_x, ..., stream)      └─ kernel<<<blockNum, nullptr, stream>>>(...)
    │                                            │
    └─ 等待流同步 ◀── 异步执行 ─── device 上每个 core 跑一个 kernel 实例
```

- **Host 侧** `run_kernel`：只做"编排"——读元信息、算分块、用 `<<<>>>` 把任务发射到 NPU 流（stream），不碰数据搬运。
- **Device 侧** `__global__` kernel：每个 AI core 执行同一份代码，通过 `GetBlockIdx()` 拿到自己的块号，只处理自己负责的那一段数据。

### 2.2 内存层级与三类 Tensor

| 名字 | 位置 | 说明 |
|---|---|---|
| `GlobalTensor<T>` (xGm/yGm) | **GM**（片外大内存） | 输入/输出数据所在，所有核共享，只能 `DataCopy` 进出 |
| `LocalTensor<T>` (xLocal/yLocal) | **UB**（片上缓存，2201 共 192KB） | 计算真正发生的地方，每个核私有 |
| `TPipe` + `TQue` | 管理 UB 里的队列 | 用队列把"搬进来 / 算 / 搬出去"串成流水 |

### 2.3 经典三段式：CopyIn → Compute → CopyOut

这是 Ascend C 所有算子（Add/Mul/……）的统一骨架：

```
CopyIn(tile)      DataCopy(GM → UB)，入队 inQueue
    ↓
Compute(tile)     出队 inQueue → 用向量算子算 → 结果入队 outQueue
    ↓
CopyOut(tile)     出队 outQueue → DataCopy(UB → GM)
```

本项目的 `KernelGeluHalf` / `KernelGeluFloat` 都是这个骨架 + 两处增强：**双缓冲**（队列深度 2）与**软件流水**（见 §四）。

### 2.4 TQue 的三个位置（本项目用到）

- `TPosition::VECIN` —— 输入队列（GM → UB）。
- `TPosition::VECOUT` —— 输出队列（UB → GM）。
- `TPosition::VECCALC` —— 纯计算用中间 buffer（fp32 的 erf 需要 4 个临时 tensor）。

> 注意：本 CANN 版本队列模板参数是 `TQue<TPosition::VECIN, 深度>`（旧版教程里的 `QuePosition` 已废弃）。

---

## 三、代码逐步解析

> 以下按 **执行顺序** 讲解每个函数的每一步，并标注当时的队列/buffer 状态。

### 3.1 `run_kernel` —— host 侧启动

```cpp
extern "C" void run_kernel(GM_ADDR input_x, const TensorGroupInfo& info_input_x,
    GM_ADDR output, const TensorGroupInfo& info_output,
    int64_t availableCoreNum, aclrtStream stream)
```

**第 1 步：解析元信息，求元素总数**
```cpp
const TensorInfo& inInfo = info_input_x.tensors[0];   // 取输入张量 0
int64_t total = 1;
for (d...) total *= inInfo.shape[d];                   // 所有维度相乘
```
`total` 就是这组数据的总元素数（比如 32、10000、100000）。

**第 2 步：按 dtype 分发**
- `inInfo.dtype == 0`（fp32）→ `gelu_custom_float`
- 其他（fp16）→ `gelu_custom`

**第 3 步（fp32 分支）：动态 tileNum + 分块**

```cpp
uint32_t tileNum = TILE_NUM;                          // 默认 2048
if (count > (uint32_t)availableCoreNum * TILE_NUM) {  // 数据超过 核数×2048
    tileNum = TILE_NUM * 2;                           // 改用 4096（大数据）
}
uint32_t totalTiles  = (count + tileNum - 1) / tileNum;   // 需要的 tile 总数
uint32_t blockNum    = min(availableCoreNum, totalTiles); // 最多开这么多核
uint32_t tilesPerBlock = (totalTiles + blockNum - 1) / blockNum; // 每核分几个 tile
uint32_t perBlock    = tilesPerBlock * tileNum;            // 每核处理元素数（32B 对齐）
gelu_custom_float<<<blockNum, nullptr, stream>>>(input_x, output, count, perBlock, tileNum);
```
关键设计：
- **`perBlock` 是 `tileNum` 的整数倍** → 每个核的 GM 偏移天然 32B 对齐（`DataCopy` 要求，否则数据错位）。
- `totalTiles > blockNum` 时，多余的 tile 均匀分到各核；末尾 block 若 `offset ≥ count` 自动退出。

**第 4 步（fp16 分支）**：逻辑同上，`tileNum` 固定 2048。

### 3.2 `gelu_custom_float` / `gelu_custom` —— device kernel 入口

```cpp
extern "C" __global__ __vector__ void gelu_custom_float(
    GM_ADDR input_x, GM_ADDR output, uint32_t totalCount, uint32_t perBlock, uint32_t tileNum)
```

**第 1 步：算自己（本 core）负责的段**
```cpp
uint32_t offset = (uint32_t)GetBlockIdx() * perBlock;   // 本核起点（元素下标）
if (offset >= totalCount) return;                       // 超出范围 → 空闲核，直接返回
uint32_t count = min(perBlock, totalCount - offset);    // 本核元素数（最后一个核可能不满）
```
`GetBlockIdx()` 由硬件提供：0,1,2,…,blockNum-1。例如 `blockNum=24`，core 7 负责 `[7*perBlock, 7*perBlock+perBlock)`。

**第 2 步：构造算子对象并驱动**
```cpp
KernelGeluFloat op;                 // 栈上对象（每核一份）
op.Init(input_x, output, offset, count, tileNum);  // 绑定 GM、建队列、记偏移
op.Process();                       // 三段式主循环
```
> 注意：`offset`（本核在全局数据里的起点）存进对象里，后续 `CopyIn/CopyOut` 都用 `xGm[offset + tileOffset]` 定位，实现"每核只搬自己那段"。

---

### 3.3 `KernelGeluHalf`（fp16，全程函数步进）

#### 3.3.1 `Init` —— 建立数据通道

| 步骤 | 动作 | 状态 |
|---|---|---|
| 1 | `xGm.SetGlobalBuffer((__gm__ half*)x)` | GlobalTensor 绑定输入 GM 地址 |
| 2 | `yGm.SetGlobalBuffer((__gm__ half*)y)` | GlobalTensor 绑定输出 GM 地址 |
| 3 | `pipe.InitBuffer(inQueueX, 2, TILE_NUM*sizeof(half))` | 在 UB 划 2 个 buffer 给输入队列（深度 2 = 双缓冲） |
| 4 | `pipe.InitBuffer(outQueueY, 2, TILE_NUM*sizeof(half))` | 划 2 个 buffer 给输出队列 |
| 5 | 记录 `offset`、`totalCount` | 后续分块定位用 |

#### 3.3.2 `Process` —— 软件流水主循环（逐步）

```
loopCount = ceil(totalCount / TILE_NUM)

预取：CopyIn(tile 0)                  ← 先搬第一个 tile
loop i = 0 .. loopCount-1:
    ① Compute(tile i)                 ← 算第 i 块
    ② if (i+1 < loopCount) CopyIn(tile i+1)   ← 预取下一块（异步，和③的 DataCopy 并行）
    ③ CopyOut(tile i)                 ← 写回第 i 块
```
以 `loopCount=3` 为例，队列状态推演（inQueue 深度 2 = buf A/B，outQueue 深度 2 = buf X/Y）：

| 时刻 | 动作 | inQueue | outQueue | 引擎并行 |
|---|---|---|---|---|
| 0 | CopyIn(0) | [A: tile0] | | MTE2 搬 GM→UB |
| 1 | Compute(0) | A 出队计算 | [X: res0] | V 向量计算 |
| 2 | CopyIn(1) | [B: tile1] | | MTE2 搬 tile1 |
| 3 | CopyOut(0) | | X 出队搬出 | **MTE3 搬 res0 ↔ MTE2 搬 tile1 并行** |
| 4 | Compute(1) | B 出队 | [Y: res1] | V |
| 5 | CopyIn(2) | [A: tile2] | | MTE2 |
| 6 | CopyOut(1) | | Y 出队 | MTE3 |
| 7 | Compute(2) | A 出队 | [X: res2] | V |
| 8 | CopyOut(2) | | X 出队 | MTE3 |

→ **收益**：本来"搬入→算→搬出"是串行（每一步等上一步完成），软件流水让下一块的搬入和上一块的搬出重叠（MTE2/MTE3 是不同硬件引擎，真并行）。这就是 Ascend C 标准双缓冲 + 手动软件流水的经典写法。

#### 3.3.3 `CopyIn` / `Compute` / `CopyOut`

```cpp
CopyIn(tileOffset, count):
    xLocal = inQueueX.AllocTensor<half>();   // 拿一个空闲 UB buffer（A/B 轮换）
    DataCopy(xLocal, xGm[offset+tileOffset], count);  // GM→UB，异步发起
    inQueueX.EnQue(xLocal);                  // 标记"这块数据可用"

Compute(count):
    xLocal = inQueueX.DeQue<half>();         // 取可用数据（等待其 DataCopy 完成）
    yLocal = outQueueY.AllocTensor<half>();
    Gelu<half>(yLocal, xLocal, count);       // 向量 GELU（tanh 近似）
    outQueueY.EnQue(yLocal);                 // 结果入输出队列
    inQueueX.FreeTensor(xLocal);             // 释放输入 buffer → 可被下一次 CopyIn 复用

CopyOut(tileOffset, count):
    yLocal = outQueueY.DeQue<half>();
    DataCopy(yGm[offset+tileOffset], yLocal, count);  // UB→GM
    outQueueY.FreeTensor(yLocal);
```

`EnQue/DeQue/AllocTensor/FreeTensor` 是 Ascend C 的**队列同步原语**：`DeQue` 会阻塞直到对应 `EnQue` 的数据就绪，`FreeTensor` 释放 buffer 供下一轮复用——这正是双缓冲能安全轮换的原因。

---

### 3.4 `KernelGeluFloat`（fp32，手写精确 erf）

#### 3.4.1 `Init` —— 比 fp16 多一个计算队列

| 步骤 | 动作 |
|---|---|
| 1-2 | `xGm`/`yGm` 绑定 GM |
| 3 | `InitBuffer(inQueueX, 2, tileNum*sizeof(float))`（tileNum 是运行时 2048/4096） |
| 4 | `InitBuffer(calcQueue, 4, tileNum*sizeof(float))` ← **4 个计算中间 buffer** |
| 5 | `InitBuffer(outQueueY, 2, tileNum*sizeof(float))` |
| 6 | 记录 `offset/totalCount/tileNum` |

#### 3.4.2 `Process` —— 与 fp16 完全相同的软件流水（只是 `TILE_NUM`→`tileNum`）

#### 3.4.3 `Compute` —— 精确 erf 逐步推演

> 用到的 4 个中间 buffer：`b0=x²`、`b1=t`、`b2=P(→erf)`、`b3=Q`。
> 所有算子都是 **Level 2（count 模式）**，自动处理 mask，支持非 32 对齐尾部。

**① 准备 t 并 clip**（对应 `erf` 的定义域限制）
```cpp
b1 = x × 0.70710678        // t = x/√2          (Muls)
b1 = max(b1, -3.92)        // 下限 clip         (Maxs)
b1 = min(b1,  3.92)        // 上限 clip         (Mins)
```

**② 计算 x²**
```cpp
b0 = b1 × b1               // x²                (Mul)
```

**③ P(x) 多项式（Horner 嵌套，6 阶×t）**
```cpp
b2 = b0 × 0.053443748819          // b2 = p5·x²
b2 = b2 + 7.5517016694            //    + p4
b2 = b2 × b0                      //    ·x²
b2 = b2 + 101.62808918            //    + p3
b2 = b2 × b0
b2 = b2 + 1393.8061484
b2 = b2 × b0
b2 = b2 + 5063.7915060
b2 = b2 × b0
b2 = b2 + 29639.384698            //    + p0
b2 = b2 × b1                      //    ·t  → P(x)
```

**④ Q(x) 多项式（5 阶）**
```cpp
b3 = b0 + 31.212858877            // x²+q4
b3 = b3 × b0
b3 = b3 + 398.56963806
b3 = b3 × b0
b3 = b3 + 3023.1248150
b3 = b3 × b0
b3 = b3 + 13243.365831
b3 = b3 × b0
b3 = b3 + 26267.224157            // → Q(x)
```

**⑤ erf = P / Q**（必须用硬件 `Div`）
```cpp
b2 = b2 / b3                      // erf(t)       (Div)
```
> ⚠️ 曾尝试 `Reciprocal(b3)+Mul` 替代，但 2201 的 `Reciprocal<float>` 是快速近似，误差 >1e-4，实测 32% 元素错——**精确 erf 必须用 `Div`**。

**⑥ 拼装最终 GELU**
```cpp
b2 = b2 + 1.0                     // 1+erf        (Adds)
y  = x × 0.5                      // 0.5·x        (Muls)
y  = y × b2                       // 0.5·x·(1+erf) (Mul)
```

**⑦ 释放 4 个中间 buffer、入队结果**
```cpp
calcQueue.FreeTensor(b0..b3);     // 归还计算 buffer
outQueueY.EnQue(y);
inQueueX.FreeTensor(x);
```

> 数值：fp32 下最大绝对误差 ~1e-6（`Div` 1 ULP），对 1e-4 的精度要求余量约 100 倍。

---

## 四、双缓冲与软件流水深度解析

### 4.1 为什么队列深度是 2？

深度 2 = 一块在"正在用"、一块在"正在搬/空闲"。`AllocTensor` 拿空闲块、`DeQue` 拿就绪块、`FreeTensor` 归还——靠这套原语实现 A/B 轮换，既避免重复等待，又不会覆盖正在用的数据。深度 3 可做更深流水，但本项目实测收益很小（瓶颈在 Compute 而非 DataCopy）。

### 4.2 为什么多核偏移必须 32B 对齐？

`DataCopy` 以 32B 为搬运粒度。若某核的起点 `offset` 不是 32B 整数倍（half 16 个 / float 8 个），搬运会从"半块"开始，**数据错位**。本项目曾因此出现 0.03%~0.35% 边界错误。解法：`perBlock = tilesPerBlock × tileNum`，让每个核的起点都是 `tileNum` 的整数倍。

### 4.3 为什么 fp32 要动态 tileNum？

- **小数据**（`totalTiles ≤ 核数`）：2048 让每个核只分 1 个 tile，核数= tile 数，并行度最高。
- **大数据**（`totalTiles > 核数`）：4096 减少每个核的循环次数、并让 tile 数更接近核数的整数倍（负载均衡），实测测试点 5 从 19.18μs → 17.38μs。
- 但 4096 对中等数据（几千~一万）会砍并行核数（5 核→3 核），反而变慢 → 所以按 `count > availableCoreNum×2048` 动态切换。

---

## 五、性能优化历程与踩坑

| 版本 | 问题 | 结果 |
|---|---|---|
| v1 fp16-only | 只实现 fp16，fp32 按 fp16 读 | 2/5（WA 99.8%） |
| v2 Gelu 多 dtype | dtype 已分发，但 tanh 近似 fp32 精度不足 | 2 Pass + 3 WA(5%) |
| v3 精确 erf（Erf API） | `AscendC::Erf` 高级 API 真机全量错误 | 0/5（100%） |
| v4 手写 erf + Div | fp16 `Gelu<half>` + fp32 手写精确 erf | **5/5 Pass** |
| v5 多核分块 | 分块 offset 非 32B 对齐 → 边界错位 | 0.03%~0.35% 错 |
| v6 tile 粒度对齐 | `perBlock = tilesPerBlock×TILE_NUM` 保证 32B 对齐 | 5/5 Pass，87.2 分 |
| v7 TILE_NUM=4096 | 中等数据并行度下降 | 负优化（回退） |
| v8 软件流水 | DataCopy 重叠 | 测试点5 收益 ~0.03μs |
| v9 Reciprocal+Mul | `Reciprocal<float>` 精度不足 1e-4 | 32% 错（回退） |
| **v10 动态 TILE_NUM** | fp32 大数据 4096 / 小数据 2048 | **84.68 分 / 第 1** |

### 调优过程

1. **dtype 分发**：评测约定 `0=fp32, 1=fp16, 2=bf16`，只实现一种会全错。
2. **`AscendC::Erf` 高级 API 不可用**：direct invocation + 2201 真机全量输出错误（疑 mask/workspace 交互），必须 basic API 手写。
3. **`Reciprocal<float>` 精度不足**：fast 近似误差 > 1e-4，精确 erf 必须 `Div`。
4. **32B 对齐**：多核分块每个 block 偏移必须是 32B 整数倍。
5. **2201 UB=192KB**：不是 256KB，tile 和 buffer 深度要留足空间。
6. **`TILE_NUM` 需权衡**：过大伤中等数据并行度，过小增大循环开销，按数据量动态选择。

### 最终性能（对比第 2 名 EigenLord）

| 测试点 | 我 | EigenLord |
|---|---|---|
| 1 | **2.96μs** | 3.64μs |
| 2 | 3.96μs | **3.92μs** |
| 3 | **5.66μs** | 5.98μs |
| 4 | **3.97μs** | 4.37μs |
| 5 | 19.18μs | **18.18μs** |

---

## 六、构建与运行

### 6.1 环境要求

- CANN Toolkit（9.2.0-beta.1），昇腾 910B（dav-2201）。
- `export ASCEND_HOME_PATH=... && source ${ASCEND_HOME_PATH}/set_env.sh`
- 本机若无 NPU 硬件，编译可过但运行 `aclInit` 失败（`ACL_ERROR_INTERNAL_ERROR`），需真机/评测环境。
- 可直接在cannjudge找到题目，提交文件，得到结果

  
### 6.2 全流程

```bash
bash run.sh   # cmake && make → 生成数据 → 运行 → verify_result.py 校验
```

### 6.3 单独编译

```bash
cd build && cmake .. && make -j4
```

---

## 七、备注

- 本文档对应 `kernel.asc` 的 **v10 动态 TILE_NUM** 版本（84.68 分 / 第 1）。
- 继续优化方向：fp16 也做动态 TILE_NUM（测试点 4 用 4096 实测更快 0.5μs）；fp32 `Compute` 指令级优化（收益小且需真机反复验证）。

