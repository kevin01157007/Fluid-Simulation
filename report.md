# HW3 Report — Croc's Waterworks: 2D PIC/FLIP Fluid Simulation

**Student Name:** 邱顯成
**Student ID:** 315554051  

> Replace all placeholder text before submission. Keep the report concise, but
> include enough explanation and evidence for the TA to understand and verify
> your implementation.

---

## 1. Implementation

### 1.1 TODO 1 — Particle Relaxation

Describe how you implemented `particleRelaxation()`.

Please discuss:

- how particles are assigned to the relaxation lookup grid,
- why only a 3 × 3 neighborhood needs to be searched,
- how overlapping particles are detected,
- how the symmetric position correction is computed,
- how you avoid processing the same particle pair twice,
- how you handle the exact-overlap / zero-distance case,
- how terrain collision is handled after relaxation.

**Implementation summary:**
透過 Spatial Hashing 概念，將每個粒子の二維坐標除以儲存格尺寸 `relaxation_cell_dim`，再利用 `std::floor` 向下取整數得到對應的 col 與 row。透過 `std::clamp` 限制在合法邊界內後，將粒子索引 p 推入對應的 `relaxation_cell_particle_ids` 陣列中。

```cpp
for (int p = 0; p < num_particles; ++p)
{
    const int col = std::clamp(
        static_cast<int>(std::floor(particle_pos[p].x() / relaxation_cell_dim)),
        0, relaxation_cell_cols - 1);
    const int row = std::clamp(
        static_cast<int>(std::floor(particle_pos[p].y() / relaxation_cell_dim)),
        0, relaxation_cell_rows - 1);

    relaxation_cell_particle_ids[row][col].push_back(p);
}
```
接著，為避免不必要的 $N^2$ 全局比對，演算法僅需針對當前儲存格及其周圍的 $3 \times 3$ neighborhood 進行檢索。在檢索過程中，透過條件 `if (q <= p) continue;` 略過自我與重複組合以避免重複處理相同粒子對。透過計算兩粒子間的距離，若小於兩倍半徑 (`min_distance = 2 * particle_radius`) 則判定為重疊；若距離趨近於零 (`dist < EPS`) 則給予微小位移以防止除以零的錯誤。當檢測到重疊時，會計算出大小相等、方向相反的 symmetric position correction 向量，分別從兩粒子的位置扣除與疊加。

```cpp
if (dist < EPS)
{
    correction = Eigen::Vector2f(0.001f, 0.001f);
}
else
{
    correction = 0.5f * (min_distance - dist) * (d / dist);
}
```
最後，由於 relaxation 可能會將粒子推入實體牆壁，因此在每次迭代結束後，必須針對所有粒子呼叫 `resolveParticleTerrainCollision` 以正確處理 terrain collision。

---

### 1.2 TODO 2 — PIC/FLIP Velocity Transfer

Describe your implementation of `isFaceVelocityValid()` and
`transferVelocities(bool to_cell)`.

Please discuss:

- the staggered MAC-grid layout,
- why Vx and Vy use different sampling offsets,
- how the four bilinear interpolation weights are computed,
- Particle → Grid accumulation and weight normalization,
- when and why `prev_cell_velocities` is saved,
- how valid / invalid MAC faces are determined,
- how PIC velocity is computed,
- how FLIP velocity is computed,
- how `flip_ratio` blends PIC and FLIP.

**Implementation summary:**  
速度轉移分成 Particle → Grid 與 Grid → Particle 兩個方向。網格採用 staggered MAC-grid：`cell_velocities[row][col].x()` 儲存該 cell 左側垂直面上的水平速度 Vx，`.y()` 儲存下側水平面上的垂直速度 Vy。因此 Vx 與 Vy 的取樣位置不同：以 cell 尺寸 `cell_dim` 為尺度 ( 半格的實際距離為 $0.5 \times cell\_dim$ )，Vx 的偏移是 `(0, 0.5)`，Vy 的偏移是 `(0.5, 0)`。這樣可讓速度分量取樣於各自方向的 cell face，而不是都取在 cell center。

對每個粒子及速度分量，先依該分量的偏移量換算到對應的取樣網格，再用 `floor` 找到左下方的基準樣本 `(base_col, base_row)`。另外三個樣本依序為右下、右上與左上。令粒子相對於基準樣本的局部座標為 `tx = clamp(dx / cell_dim, 0, 1)`、`ty = clamp(dy / cell_dim, 0, 1)`，四個雙線性權重依序為：

```text
w1 = (1 - tx) * (1 - ty)    // 左下
w2 = tx * (1 - ty)          // 右下
w3 = tx * ty                // 右上
w4 = (1 - tx) * ty          // 左上
```

以下程式依速度分量套用不同的半格偏移，並計算四個雙線性權重：

```cpp
float fx = pos.x();
float fy = pos.y();
if (component == 0)
    fy -= 0.5f * cell_dim; // Vx: (0, 0.5)
else
    fx -= 0.5f * cell_dim; // Vy: (0.5, 0)

int base_col = static_cast<int>(std::floor(fx / cell_dim));
int base_row = static_cast<int>(std::floor(fy / cell_dim));
float x_node = base_col * cell_dim;
float y_node = base_row * cell_dim;
if (component == 0)
    y_node += 0.5f * cell_dim;
else
    x_node += 0.5f * cell_dim;

float tx = clampf((pos.x() - x_node) / cell_dim, 0.0f, 1.0f);
float ty = clampf((pos.y() - y_node) / cell_dim, 0.0f, 1.0f);

weight[0] = (1.0f - tx) * (1.0f - ty);
weight[1] = tx * (1.0f - ty);
weight[2] = tx * ty;
weight[3] = (1.0f - tx) * ty;
```

進行 Particle → Grid 時，先呼叫 `applyTerrainTypes()` 更新 SOLID 與 FLUID 標記，並將網格速度歸零。每個粒子把自己的對應速度分量乘上四個權重，累加至周圍有效範圍內的網格樣本；同時累加各樣本的權重總和。所有粒子處理完後，各有貢獻的樣本以速度加權總和除以權重總和，形成加權平均；沒有貢獻的分量設為零。

正規化後，呼叫 `solveIncompressibility()` 投影之前，將整個速度場複製到 `prev_cell_velocities`。投影會改變目前網格速度，而 FLIP 需要比較投影後與投影前的速度差，因此必須在這個時間點保存原始網格速度。

```cpp
if (component == 0)
{
    cell_velocities[r][c].x() += particle_vel[p].x() * weight[k];
    weight_sum[r][c].x() += weight[k];
}
else
{
    cell_velocities[r][c].y() += particle_vel[p].y() * weight[k];
    weight_sum[r][c].y() += weight[k];
}

if (weight_sum[r][c].x() > EPS)
    cell_velocities[r][c].x() /= weight_sum[r][c].x();
else
    cell_velocities[r][c].x() = 0.0f;

if (weight_sum[r][c].y() > EPS)
    cell_velocities[r][c].y() /= weight_sum[r][c].y();
else
    cell_velocities[r][c].y() = 0.0f;

prev_cell_velocities = cell_velocities;
```

Grid → Particle 插值時，`isFaceVelocityValid()` 先檢查樣本索引是否在該分量的 MAC 網格範圍內，再判斷面相鄰的 cell 是否至少有一個是 `FLUID`。Vx 面檢查左右相鄰 cell `(row, col - 1)` 與 `(row, col)`；Vy 面檢查下上相鄰 cell `(row - 1, col)` 與 `(row, col)`。超出網格範圍、或相鄰 cell 都不是 FLUID 的面不參與插值。剩餘有效樣本的權重會重新加總並正規化，避免無效面使插值結果偏小。
```text
      Vy 面
   ─────────
   |       |
Vx | cell  | Vx
面 |       | 面
   ─────────
      Vy 面
```

```cpp
bool has_fluid = false;
if (component == 0) // Vx: (row, col - 1) 與 (row, col)
{
    if (col - 1 >= 0 && cell_types[row][col - 1] == CellType::FLUID)
        has_fluid = true;
    if (col < cell_cols && cell_types[row][col] == CellType::FLUID)
        has_fluid = true;
}
else if (component == 1) // Vy: (row - 1, col) 與 (row, col)
{
    if (row - 1 >= 0 && cell_types[row - 1][col] == CellType::FLUID)
        has_fluid = true;
    if (row < cell_rows && cell_types[row][col] == CellType::FLUID)
        has_fluid = true;
}
```

PIC 速度是有效樣本目前網格速度的加權平均：$v_{pic} = \frac{\sum w_i v_i}{\sum w_i}$。FLIP 速度則保留粒子原有速度，再加上網格投影造成的加權速度變化：$v_{flip} = v_{particle\_old} + \frac{\sum w_i (v_i - v_{i\_previous})}{\sum w_i}$，其中 $v_{i\_previous}$ 取自 `prev_cell_velocities`。最後依 `flip_ratio` 混合兩者：$v = (1 - \alpha) v_{pic} + \alpha v_{flip}$；ratio 為 0 時完全採用 PIC，ratio 為 1 時完全採用 FLIP。若沒有任何有效權重，不更新該粒子的速度分量。

```cpp
pic_val /= total_weight;
flip_diff /= total_weight;
float flip_val = old_vel + flip_diff;
float blended_vel = (1.0f - flip_ratio) * pic_val + flip_ratio * flip_val;
```

---

### 1.3 TODO 3 — Cell Density Estimation

Describe your implementation of `updateDensity()`.

Please discuss:

- why density is sampled at cell centers,
- how a particle contributes to four surrounding density samples,
- how SOLID cells are handled,
- how `particle_rest_density` is initialized,
- why the rest density is computed only once.

**Implementation summary:**  
每一幀開始時，先將所有 `cell_densities` 歸零，再將粒子密度分配至周圍網格。由於密度代表 cell 內粒子的分布，取樣位置使用 cell center，偏移量為 `(0.5, 0.5)` 個 cell；這與取樣在 cell face 上的 MAC 速度不同。程式從粒子位置的 x、y 座標各減去 $0.5 \times cell\_dim$，換算到 cell-center 樣本網格，並用 `floor` 找到左下方基準樣本。

接著找出周圍四個 cell-center 樣本，依粒子相對於基準樣本的局部座標 `tx`、`ty` 計算雙線性權重：

```cpp
float fx = pos.x() - 0.5f * cell_dim;
float fy = pos.y() - 0.5f * cell_dim;
int base_col = static_cast<int>(std::floor(fx / cell_dim));
int base_row = static_cast<int>(std::floor(fy / cell_dim));

float x_node = (base_col + 0.5f) * cell_dim;
float y_node = (base_row + 0.5f) * cell_dim;
float tx = clampf((pos.x() - x_node) / cell_dim, 0.0f, 1.0f);
float ty = clampf((pos.y() - y_node) / cell_dim, 0.0f, 1.0f);

weight[0] = (1.0f - tx) * (1.0f - ty);
weight[1] = tx * (1.0f - ty);
weight[2] = tx * ty;
weight[3] = (1.0f - tx) * ty;
```

對每個位於網格範圍內的樣本，若該 cell 不是 `SOLID`，就將粒子的單位貢獻乘上對應權重並累加至 `cell_densities`；因此粒子會將密度平滑分散到周圍 cell，而不只落在單一格中。`SOLID` cell 不接收密度貢獻：

```cpp
if (r >= 0 && r < cell_rows && c >= 0 && c < cell_cols)
{
    if (cell_types[r][c] != CellType::SOLID)
    {
        cell_densities[r][c] += 1.0f * weight[k];
    }
}
```

當 `particle_rest_density` 尚未初始化（值小於或等於零）時，程式只在第一次密度更新後，計算所有 `FLUID` cell 的密度平均值，作為粒子群的參考 `particle_rest_density`。若當時沒有任何 `FLUID` cell，則使用 `1.0` 來防止除以零會出錯。`particle_rest_density` 只初始化一次，是為了保留初始粒子排列所建立的參考值；若每一幀都重新計算，參考值會跟著當前壓縮或膨脹變動，削弱後續密度修正的依據。

```cpp
if (particle_rest_density <= 0.0f)
{
    float total_density = 0.0f;
    int fluid_cell_count = 0;

    for (int r = 0; r < cell_rows; ++r)
    {
        for (int c = 0; c < cell_cols; ++c)
        {
            if (cell_types[r][c] == CellType::FLUID)
            {
                total_density += cell_densities[r][c];
                fluid_cell_count++;
            }
        }
    }

    if (fluid_cell_count > 0)
        particle_rest_density = total_density / static_cast<float>(fluid_cell_count);
    else
        particle_rest_density = 1.0f;
}
```

---

### 1.4 TODO 4 — Incompressibility Solver

Describe your implementation of `solveIncompressibility()`.

Please discuss:

- how divergence is computed from the four MAC-face velocities,
- how SOLID neighbors affect the number of available flow directions,
- how the divergence correction is distributed,
- how `over_relaxation` is used,
- how density correction is added when a cell is over-compressed,
- why the solver is repeated for multiple iterations.

**Implementation summary:**  
不可壓縮性求解器會依設定的 `iterations` 重複掃描網格，逐步修正流體速度，使 `FLUID` cell 的散度趨近於零。每次掃描只處理內部 cell（row、col 從 1 到邊界前一格），以確保讀取相鄰 cell 與 MAC 面速度時索引有效；非 `FLUID` cell 會直接略過。

對每個 `FLUID` cell，先檢查左右上下相鄰 cell 是否不是 `SOLID`，並將可用方向數量加總。與 `SOLID` 相鄰的方向不可流動，因此不納入修正；若四個方向都不可用，就略過該 cell，避免除以零。接著依 MAC-grid 的面速度計算散度：

```text
left   = Vx(row,     col)
right  = Vx(row,     col + 1)
bottom = Vy(row,     col)
top    = Vy(row + 1, col)

divergence = top + right - left - bottom
```

若啟用 `density_correction`，且 cell 密度高於 `particle_rest_density`，就將兩者差值視為正壓縮量，並從散度扣除 `stiffness_coefficient * compression`，加入促使流體膨脹的修正；未超過靜止密度時不套用此項。之後將散度乘上 `over_relaxation`，再除以可用方向數量，得到分配到各面的修正量：

```text
correction = over_relaxation * divergence / available_direction_count
```

最後只更新可用方向的面速度：左、下速度加上修正量，右、上速度扣除修正量，藉此降低散度。重複多次迭代，讓先前更新過的速度在後續掃描中繼續調整，逐步逼近不可壓縮狀態；`over_relaxation` 則用來調整每次修正的幅度、加速收斂。

---

## 2. Results & Discussion

Perform the following **three required experiments**.

For each experiment:

- keep all other simulation parameters at their default values,
- change only the specified parameter,
- run the same scene under each setting,
- include screenshots if they help illustrate the difference,
- describe the visible behavior and explain why the parameter causes that effect.

### 2.1 Experiment 1 — FLIP Ratio

Test the following values:

- `0.0`
- `0.5`
- `0.95`

**Other parameters:** kept at default values.

| FLIP Ratio | Observation |
| --- | --- |
| 0.0 | 動能消散極快，流體表現出高度黏滯性（類似蜂蜜），缺乏水花與動態細節。 |
| 0.5 | 流體兼具穩定性與適度的動態表現，能維持合理的流動感與部分水花。 |
| 0.95 | 高度保留動能，水流極為活潑且容易產生豐富的水花與波動。 |

**Discussion:**  
Compare the motion produced by lower and higher FLIP ratios. Discuss effects such
as smoothness, damping, preservation of motion, splashing, and numerical noise.

當 FLIP ratio 為 0 時，velocities 會在每個 time step 從 grid 完全重新進行內插。這會引入強烈的數值耗散，導致 fluid 快速失去 kinetic energy，表現得高黏滯性。

相反地，當 FLIP ratio 接近 1（幾乎全是 FLIP）時，simulation 會將來自 grid 的 velocity changes（delta-velocities）累積到 particles 上。這能保留 particles 的 momentum、vortex dynamics 以及 high-frequency kinetic energy，進而產生生動且逼真的 splashes。
不過，若使用純 FLIP 或極高的 FLIP ratio，當 velocity projection errors 在 frames 之間累積時，有時會引入輕微的 numerical noise 或 instability。
一個平衡的 ratio（大約 0.5 到 0.95 之間）能在 stability 與 energy conservation 之間取得理想的平衡。

---

### 2.2 Experiment 2 — Solver Iterations

Test the following values:

- `1`
- `3`
- `8`

**Other parameters:** kept at default values.

| Solver Iterations | Observation |
| --- | --- |
| 1 | 流體壓縮嚴重，無法有效滿足 Incompressibility，體積無法保持。 |
| 3 | 體積與不可壓縮性達到可接受的水準。 |
| 8 | 散度被嚴格壓制，流體表現出極佳的體積保持能力與穩定性 |

**Discussion:**  
Compare how the number of incompressibility-solver iterations affects fluid
compression, volume preservation, visual stability, and computational cost.

Incompressibility solver 採用迭代 Gauss-Seidel projection 來消除 MAC 網格上的速度散度。

單次迭代（1）不足以使壓力/散度收斂，導致高散度、體積損失，甚至可能錯誤。default（3次迭代）在即時互動性和精確度之間取得了良好的平衡，足以滿足互動遊戲的不可壓縮性條件。
將迭代次數增加到8次可以更嚴格地在整個網格上強制執行無散度條件，從而最大限度地保證體積守恆，但代價是每幀的 CPU 開銷會增加。


---

### 2.3 Experiment 3 — Stiffness Coefficient

Test the following values:

- `0.0`
- `1.0`
- `3.0`

**Other parameters:** kept at default values.

| Stiffness Coefficient | Observation |
| --- | --- |
| 0.0 | 粒子在受到擠壓或堆積時容易互相重疊、緊縮，無法有效維持均勻的粒子間距。 |
| 1.0 | 有效產生適度的 Expansion bias，使過密區域的粒子自然散開。 |
| 0.0 (or high, e.g., 3.0) | 粒子間距被強制拉開，流體表現出過度彈性或類似表面張力過高的異常震動 |

**Discussion:**  
Compare how the stiffness coefficient affects density correction. Discuss
particle compression, spreading behavior, stability, and any visible artifacts
when the coefficient becomes larger.

Stiffness Coefficient ($k_{stiff}$) 用於控制在 incompressibility solve 期間所加入的 density correction term 之大小。
當設為 0 時，density correction 不會啟用；solver 僅檢查 velocity divergence，允許 particles 超過其 physical rest density 緊密堆積，導致不自然的 clumping。在適中數值（1）下，會根據 cell density errors 引入平衡的 pressure-like repulsion，幫助 particles 自然散開。若調高至高值（3），排斥力會變得過於強烈，引入過大的 correction forces，使 fluid surface 呈現 jittery、springy 或 unstable 的狀態，因 particles 會彼此激烈推開。

---

## 3. Final Demo Result

Include at least one screenshot showing that your completed implementation can
successfully guide water through the level and reach the bathtub.

<figure align="center">
  <img src="assets/image.png" alt="模擬中水流已到達浴缸；截圖時收集進度為 91%。">
  <figcaption><strong>Figure 1 : Demo</strong></figcaption>
</figure>

Briefly describe whether the simulation behaves as expected and mention any
remaining visual or numerical issues you observed.

模擬大致符合預期：挖出通道後，水會進入管線並從蓮蓬頭流入浴缸。截圖時已收集 2371 / 2600 個粒子（約 91%），另有 6 個在管線中、223 個仍在場景中；繼續挖通道後，觀察到進度最高約為 100%。

## 4. Bonus / Miscellaneous (Optional)

If you implemented any creative extension or changed code outside the required
TODO sections, describe it here.

Examples:

- additional visual effects,
- new interaction / gameplay behavior,
- alternative numerical method,
- performance optimization,
- additional experiment or visualization.

**Description:**  

新增密度與速度大小兩種熱圖視圖，方便觀察流體在網格上的分布與運動狀態，且不改變模擬更新或求解流程。按鍵 4 可切換密度熱圖，按鍵 5 可切換速度熱圖，也可在 Controls 面板選擇；原有水粒子、網格與混合視圖仍可使用。

密度熱圖以每個 FLUID cell 的 cell density 著色，色階範圍為 0 至 2 倍 particle rest density；速度熱圖先將 MAC 網格相鄰面的 Vx、Vy 平均到 cell center，再以速度大小著色，色階範圍為 0 至 50。兩種熱圖皆由藍色（低值）漸變至紅色（高值），超出色階範圍的數值會限制在色階端點。密度圖可用來辨識粒子聚集或高密度區域，速度圖則呈現流速較快的區域。


**Screenshot / evidence (if applicable):**  
```text
執行程式後按 4、5 可分別查看密度與速度熱圖；提交報告時可在此附上兩種視圖的截圖。
```
<figure align="center">
  <img src="assets/densityheatmap.png" alt="替代文字">
  <figcaption><strong>Figure 2 : Density Heatmap</strong></figcaption>
</figure>

<figure align="center">
  <img src="assets/velocityheatmap.png" alt="替代文字">
  <figcaption><strong>Figure 3 : Velocity Heatmap</strong></figcaption>
</figure>