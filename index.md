---
title: GreenCodeAnalyzer
subtitle: A VSCode Extension for Static Energy Analysis in Data Science
description: A VSCode Extension for Static Energy Analysis in Data Science
---

**GreenCodeAnalyzer** is a static code analysis tool that identifies energy-inefficient patterns in Python code and suggests optimizations to improve energy consumption. By shifting energy efficiency concerns "left" (earlier) in the development process, developers can make more sustainable coding decisions from the start.

## Features

- **Static Energy Analysis**: Analyzes Python code without executing it to detect potential energy hotspots
- **Visual Code Annotations**: VS Code extension that provides visual feedback with highlighted energy smells
- **Optimization Suggestions**: Provides specific recommendations to make code more energy-efficient
- **Multiple Rule Detection**: Covers various energy-inefficient patterns common in data science and ML code

## Installation & Usage

### Installing the Extension from VS Code Marketplace

You can install the GreenCodeAnalyzer extension directly from the **VS Code Marketplace**:

1. Open VS Code
2. Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window or press `Ctrl+Shift+X`
3. Search for "GreenCodeAnalyzer" and click on the install button.

Alternatively, you can install it from the [VS Code Marketplace website](https://marketplace.visualstudio.com/items?itemName=KevinHoxha.GreenCodeAnalyzer).

### Using the Extension

Once installed, you can analyze your Python code for energy inefficiencies:

1. Open a Python file in VS Code
2. Use one of the following methods to run the analyzer:
   - Press `Ctrl+Shift+P` to open the Command Palette, then type and select "GreenCodeAnalyzer: Run Analyzer"
   - Right-click in the editor and select "Run GreenCodeAnalyzer" from the context menu

The analysis results will appear as decorations in your code editor, highlighting potential energy inefficiencies with suggestions for improvement.

To clear the analysis markers:

- Press `Ctrl+Shift+P` and select "GreenCodeAnalyzer: Clear Gutters"
- Or right-click and select "Clear GreenCodeAnalyzer Gutters"

#### Interpreting Results

Each detected code smell includes:

- A description of the energy inefficiency
- A specific recommendation for optimization

By integrating seamlessly with the VS Code interface, this extension ensures that developers can quickly identify and fix inefficient code without leaving their workflow.

## Supported Rules

| Rule    | Description | Libraries  | Impact     | Optimization | Source |
| ------- | ----------- | ---------- | -----------| ------------ | ------ |
| **Batch Matrix Multiplication**  | When performing multiple matrix multiplications on a batch of matrices, use optimized batch operations rather than separate operations in loops. | `NumPy` `PyTorch` `TensorFlow` | GPUs thrive on parallel operations over large batches. Small, sequential operations waste cycles and keep the hardware active longer than necessary. Instead, batch matrix multiplication leverages vectorized execution.  | **NumPy**: <br> `numpy.matmul(batch, matrices)` <br>  **PyTorch**: `torch.bmm(batch_matrices1, batch_matrices2)` <br> **TensorFlow**: `tf.linalg.matmul(batch_matrices1, batch_matrices2)` | |
| **Blocking Data Loader** | Prevent using data loading strategies that stall GPU execution (e.g., single-process or sequential data loading). | `PyTorch` | If the DataLoader is set up without sufficient concurrency (`num_workers=0`) or uses blocking I/O, the GPU may remain idle while waiting for data. Asynchronous data loading keeps the GPU busy more consistently, reducing overall epoch time and energy. | Use `num_workers > 0` in DataLoader. Consider enabling `pin_memory=True` if the data is loaded from CPU memory to GPU often. For advanced scenarios, use background threads or prefetch queues. | Azzoug, A. (n.d.). GreenPyData. GitHub. https://github.com/AghilesAzzoug/GreenPyData |
| **Broadcasting**  | Normally, when you want to perform operations like addition and multiplication, you need to ensure that the operands' shapes match. Tiling can be used to match shapes but stores intermediate results. | `TensorFlow` | Broadcasting allows us to perform implicit tiling, which makes the code shorter and more memory efficient since we don’t need to store the result of the tiling operation. |  <code>a = tf.constant([[1., 2.], [3., 4.]])<br>b = tf.constant([[1.], [2.]])<br># c = a + tf.tile(b, [1, 2]) -> Remove line<br>c = a + b</code> | Hynn, S. (n.d.). Broadcasting feature not used. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/18-broadcasting-feature-not-used/ Kiani, V. (n.d.). EffectiveTensorflow. GitHub. https://github.com/vahidk/EffectiveTensorflow?tab=readme-ov-file#broadcast |
| **Calculating Gradients** | When performing inference (i.e., forward pass without training or backpropagation), PyTorch by default tracks operations for autograd. TensorFlow will track them if specified to do so. | `PyTorch` `TensorFlow`| Autograd graph tracking increases memory usage and computational cost. Disabling it during inference leads to faster execution, lower energy consumption, and reduced VRAM usage, which is particularly beneficial for GPUs and large models. | **PyTorch**: <br> `output = model(input) # Autograd is tracking gradients` <br> `with torch.no_grad():` <br> `output = model(input) # More efficient for inference` <br> **TensorFlow**: <br> <code>with tf.GradientTape(): # Unnecessary gradient tracking in inference<br>   output = model(input)<br>@tf.function(jit_compile=True) # Efficient inference<br>def inference(input):<br>   return model(input)<br>output = inference(input)</code> | PyTorch. (n.d.). torch.no_grad. PyTorch. https://pytorch.org/docs/stable/generated/torch.no.grad.html <br> Stack Overflow. (2022, April 15). What is the purpose of with torch.no_grad? Stack Overflow. https://stackoverflow.com/questions/72504734/what-is-the-purpose-of-with-torch-no-grad |
| **Chain Indexing**  | Chain indexing refers to when using `df["one"]["two"]`, Pandas will see this operation as two events: call `df["one"]` first and then `["two"]`. | `Pandas` | Performing many calls leads to excessive memory allocations and CPU-intensive Python interpreter overhead. This can result in slow and energy-consuming code. | `df.loc[:,("one","two")]` only performs a single call | Hynn, S. (n.d.). Chain indexing. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/12-chain-indexing/ |
| **Conditional Operations** | When performing a conditional operator on an array, tensor, or dataframe inside for loops. | `NumPy` `Pandas` `PyTorch` `TensorFlow` | Doing these operations in for loops leads to inefficient branching and repeated calculations. | **NumPy**: <br> `arr = np.where(arr > 5, arr, 0)` <br> **Pandas**: <br> `df['column'].where(df['column'] > 5, 0)` <br> **PyTorch**: <br> `torch.where(tensor > 5, tensor, torch.zeros_like(tensor))` <br> **TensorFlow**: <br> `tf.where(tensor > 5, tensor, tf.zeros_like(tensor))` | Hynn, S. (n.d.). Unnecessary iteration. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/2-unnecessary-iteration/ |
| **Data Parallelization** | Refrain from wrapping models in `torch.nn.DataParallel` when `torch.nn.parallel.DistributedDataParallel` (DDP) is superior, even on a single node with multiple GPUs. | `PyTorch` | `DataParallel` uses a single process to manage multiple GPU replicas, which can result in significant overhead, especially for gradient synchronization on large `models.DistributedDataParallel` creates one process per GPU (or process group) and provides more efficient communication backends (e.g., NCCL). This typically yields better throughput and uses less CPU overhead, leading to lower energy consumption. | Wrap your model with `DistributedDataParallel` (DDP), even on a single node, rather than using `DataParallel`. | Azzoug, A. (n.d.). GreenPyData. GitHub. https://github.com/AghilesAzzoug/GreenPyData |
| **Element-Wise Operations**  | When performing an element-wise operator on an array, tensor, or dataframe inside for loops. | `NumPy` `PyTorch` `TensorFlow`| Doing these operations in for loops leads to inefficient branching and repeated calculations. | **Element-wise**: <br> `arr + 1`, `np.add(arr1, arr2)`, `df['column'] + 1`, `tensor + 1`, `torch.add(tensor1, tensor2)`, `tf.add(tensor1, tensor 2)` <br> **Mapping**: <br> `np.vectorize(func)(arr)`, `df['column'].apply(func)`, `torch.vectorized_map(func, tensor)`, `tf.map_fn(func, tensor)`, `tf.vectorized_map(func, tensor)` | Hynn, S. (n.d.). Unnecessary iteration. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/2-unnecessary-iteration/ |
| **Excessive GPU Transfers**  | Frequently moving data between CPU and GPU (e.g., calling `.cpu()` and then `.cuda()` repeatedly) without necessity. | `PyTorch` | This frequent transfer of data produces a high overhead. | Keep tensors in GPU memory throughout operations or batch transfers to minimize overhead. | Balaprakash, P., et al. (2019). Adaptive Methods for Real-Time Transfer Learning on GPUs. IEEE Transactions on Parallel and Distributed Systems. |
| **Excessive Training**  | Continuing to train a model beyond the point where validation metrics stop improving. | `PyTorch` `TensorFlow` `SciKit-Learn` | Overtraining wastes GPU/CPU cycles with diminishing returns. | Implement early stopping or define a convergence criterion to halt training once metrics plateau. | Caruana, R., et al. (2001). Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping. Advances in Neural Information Processing Systems. |
| **Filter Operations** | When performing a filter operator on an array, tensor, or dataframe inside for loops. | `NumPy` `Pandas` `PyTorch` `TensorFlow` | Boolean indexing or masking allows efficient filtering, avoiding iterative checks on each element. | **NumPy**: <br> `arr[arr > 5]`, `arr[np.logical_and(arr > 0, arr < 10)]` <br> **Pandas**: <br> `df[df['column'] > 5]` <br> **PyTorch**: <br> `tensor[tensor > 5]`, `tensor[torch.logical_and(tensor > 0, tensor < 10)]`, `torch.masked_select(tensor, tensor > 5)` <br> **TensorFlow**: <br> `tf.boolean_mask(tensor, tensor > 5)` | Hynn, S. (n.d.). Unnecessary iteration. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/2-unnecessary-iteration/ |
| **Ignoring Inplace Ops** | Failing to use the in-place variants of PyTorch operations (e.g., `add_`, `mul_`, `relu_`) leads to additional memory allocations and higher overhead. | `PyTorch` | PyTorch (and most deep learning frameworks) stores tensors and gradients in memory. Creating new tensors for every operation triggers more frequent memory allocations, which consume additional CPU/GPU cycles and can cause extra garbage collection. This overhead translates to higher energy usage. | Use in-place operations (`op_()`) where they do not break gradient flow or produce unexpected side effects. | Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems. |
| **Inefficient Caching of Common Arrays** | Recreating the same arrays or tensors (e.g., repeating `np.arange(0, n)` in a loop) instead of storing or caching them. | `NumPy` `PyTorch` `TensorFlow` | Repeated creation allocates CPU/GPU cycles and memory, increasing energy usage. | Cache repeated arrays or use partial function application to avoid the overhead of repeated creation. | Breshears, C. (2015). The Art of Concurrency: A Thread Monkey’s Guide to Writing Parallel Applications. O’Reilly Media. |
| **Inefficient Data Loader Transfer** | Refrain from using standard (pageable) CPU memory for large data loads when transferring to GPU. | `PyTorch` | When transferring data from CPU to GPU, pinned (page-locked) memory can speed up and streamline transfers in CUDA. Non-pinned memory can cause additional overhead, stalling the GPU. | Enable pin_memory=True in the PyTorch DataLoader, which can significantly reduce latency for GPU-bound training. | Azzoug, A. (n.d.). GreenPyData. GitHub. https://github.com/AghilesAzzoug/GreenPyData |
| **Inefficient Data Frame Join** | Performing repeated join operations on large DataFrames without indexing or merging strategies. | `Pandas` | Large repeated joins or merges can be extremely expensive, inflating CPU time and memory usage. | Use indices, sort-merge strategies, or carefully plan merges to reduce overhead. | McKinney, W. (2017). Python for Data Analysis. O’Reilly Media. |
| **Inefficient Iterrows** | Using `iterrows` in Pandas to manipulate data row-by-row is a frequent habit in data analysis, despite being much slower than vectorized alternatives. | `Pandas` | Row-by-row iteration incurs Python overhead, slowing execution and increasing energy use. | Use vectorized methods for data manipulation. |
| **Large Batch Size Memory Swapping** | Setting a batch size in PyTorch or TensorFlow too large for GPU memory, forcing frequent memory swaps or fallback to CPU. | `PyTorch` `TensorFlow` | Memory swapping drastically slows performance and increases energy usage. | Find an optimal batch size through experiments; use gradient accumulation if large effective batch sizes are required. | Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems. |
| **Recomputing GroupBy Results** | Calling `.groupby()` multiple times on the same data with identical keys for similar aggregated statistics. | `Pandas` | Each `.groupby()` operation is expensive; re-running the same computation consumes extra CPU cycles. | Compute all required statistics in a single pass (e.g., `agg(...)`) or store intermediate results for re-use. | McKinney, W. (2017). Python for Data Analysis. O’Reilly Media. |
| **Reduction Operations** | When performing a reduction operator on an array, tensor, or dataframe inside for loops. | `NumPy` `Pandas` `PyTorch` `TensorFlow` | Reduction operations to compute sums, means, or other aggregates are slow and have been optimized in libraries. | **NumPy**: <br> `np.sum`, `np.min`, `np.max` <br> **Pandas**: <br> `df['column'].sum()`, `df['column'].mean()`, `df.agg('sum')` <br> **PyTorch**: `torch.sum(tensor)`, `torch.mean(tensor)`, `torch.max(tensor)`, `torch.max(tensor)` <br> **TensorFlow**: `tf.reduce_sum(tensor)`, `tf.reduce_mean(tensor)`, `tf.reduce_max(tensor)` | Hynn, S. (n.d.). Unnecessary iteration. DSLinter. https://hynn01.github.io/dslinter/posts/codesmells/2-unnecessary-iteration/ Kiani, V. (n.d.). EffectiveTensorflow. GitHub. https://github.com/vahidk/EffectiveTensorflow?tab=readme-ov-file#overload |
| **Redundant Model Re-Fitting** | Continuously calling `.fit()` on the same dataset multiple times without any changes in hyperparameters or data. | `SciKit-Learn` | Each `.fit()` call recreates internal data structures, incurring CPU/memory overhead. | Re-use fitted models, or partial fit if iterative approaches are needed. | Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. |

## Known Issues

- The extension only works with **Python files**.
- Some rules may produce **false positives**, depending on the context of your code.

---

This extension is a powerful tool for developers looking to improve the efficiency of their Python code and make more sustainable software decisions.

<style>
  /* Widen and center table */
  table {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    display: block !important;
    background: #fff !important;
    word-wrap: break-word !important;  /* Ensure content wraps */
  }

  /* Table cells */
  th, td {
    padding: 5px !important;
    word-wrap: break-word !important; /* Ensure content wraps */
    overflow-wrap: break-word !important; /* Ensures long words break */
    max-width: 200px !important; /* Prevent overflow */
  }

  /* Links within table cells */
  td a {
    word-wrap: break-word !important;
    word-break: break-word !important; /* Force links to break inside cells */
    overflow-wrap: break-word !important;
  }

  /* Code in table cells */
  td code, th code {
    white-space: pre-wrap !important;  /* Wrap code lines */
    word-break: break-word !important; /* Break long words */
    display: inline-block !important;  /* Ensure code respects cell width */
    max-width: 100% !important; /* Fit within cell */
  }

   /* Align cells in the third row (Broadcasting) to the top */
   table tr:nth-child(3) td {
     vertical-align: top !important;
   }

  /* Desktop: Narrow content to 70%, table fits within */
  @media screen and (min-width: 769px) {
    .page-content, .inner, main {
      max-width: 70% !important;
      width: 70% !important;
    }
    table {
      width: 100% !important; /* Fits within 70% container */
      max-width: 100% !important;
    }
    th, td {
      padding: 5px !important;
      max-width: 200px !important; /* Prevent overflow on laptops */
    }
  }

  /* Phone: Full width for everything */
  @media screen and (max-width: 768px) {
    html, body, .page-content, .inner, main, table {
      width: 100% !important;
      max-width: 100% !important;
      padding: 2px !important;
    }
    table {
      font-size: 14px !important;
      overflow-x: auto !important;
    }
    th, td {
      padding: 3px !important;
      max-width: 150px !important; /* Smaller cap for phones */
      min-width: 40px !important;
    }
  }
</style>


<script>
  document.addEventListener("DOMContentLoaded", function() {
    var banner = document.getElementById("forkme_banner");
    if (banner) {
      banner.href = "https://github.com/ianjoshi/sustainablese-g1-green-shift-left";
      banner.textContent = "View on GitHub"; 
    }
  });
</script>
