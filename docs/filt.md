
---

通过分析代码，我们可以找到这些参数支持的滤波器类型。

### `pfilt` (金字塔滤波器)

`pfilt` 参数用于 `nsctdec` 和 `nsctrec` 函数中，控制金字塔分解阶段的滤波器类型。它最终被传递给 `atrousfilters` 函数。

在 filters.py 的 `atrousfilters` 函数中，定义了 `pfilt` 的可能取值：

1.  **`'maxflat'`**: 准紧框架滤波器（Quasi-tight frame filters）。这是代码中常见的默认值。
2.  **`'pyr'`**: 一种金字塔滤波器。
3.  **`'pyrexc'`**: 'pyr' 的一种变体（交换了部分滤波器系数）。

因此，`pfilt` 的有效取值为：`'maxflat'`, `'pyr'`, `'pyrexc'`。

### `dfilt` (方向滤波器)

`dfilt` 参数也用于 `nsctdec` 和 `nsctrec` 函数，控制方向分解阶段的滤波器类型。它最终被传递给 `dfilters` 函数。

在 filters.py 的 `dfilters` 函数中，定义了 `dfilt` 的可能取值，来源比较多样：

1.  **基于阶梯结构 (Ladder Structure) 的滤波器**:
    *   `'pkva'` (或 `'pkva12'`)
    *   `'pkva8'`
    *   `'pkva6'`
    *   `'ldtest'`

2.  **最大平坦方向滤波器 (Maximally Flat Directional Filters)**:
    *   `'dmaxflatN'`，其中 `N` 是一个从 1 到 7 的整数。例如：
        *   `'dmaxflat1'`
        *   `'dmaxflat2'`
        *   ...
        *   `'dmaxflat7'` (这是代码中常见的默认值)

3.  **基于 PyWavelets 库的一维小波滤波器**:
    代码中有一个备用逻辑，如果以上名称都不匹配，它会尝试从 `pywt` (PyWavelets) 库中加载同名的一维小波，并将其转换为二维滤波器。这意味着任何 `pywt.wavelist()` 支持的小波名称理论上都可以作为 `dfilt` 的值，例如：
    *   `'db1'`, `'db2'`, ... (Daubechies)
    *   `'sym2'`, `'sym3'`, ... (Symlets)
    *   `'coif1'`, `'coif2'`, ... (Coiflets)
    *   等等。

### 总结

| 参数 | 主要用途 | 可能的取值 |
| :--- | :--- | :--- |
| **`pfilt`** | 金字塔分解滤波器 | `'maxflat'`, `'pyr'`, `'pyrexc'` |
| **`dfilt`** | 方向分解滤波器 | - `'pkva'`, `'pkva6'`, `'pkva8'`, `'pkva12'`, `'ldtest'`<br>- `'dmaxflat1'` 到 `'dmaxflat7'`<br>- 任何 PyWavelets 库支持的小波名称 (如 `'db4'`, `'sym5'` 等) |
</assistant>