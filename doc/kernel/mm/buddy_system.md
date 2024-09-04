## 说明
`buddy system` 可以说是内存管理的核心，首先先要明确几个概念：
* `numa(Non Uniform Memory Access)` 是一种内存和`cpu`的关系设计的方案。其实现了每个`cpu(可以是物理cpu,也可以是多个core的集合)`有自己独立的内存群
  `numa` 中每个cpu都可以访问物理内存，但是访问自己的内存群的速度将会远远大于访问其他`cpu`的内存群。这在一定程度上也降低了`cpu`之间的内存资源竞争的消耗，即降低了同一内存
  不同cpu同时访问的概率
* `zone` 内核的物理内存是分区管理的，这是各逻辑概念仅是为了方便管理
  <img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/zone_type.webp />
  1. `amd64`架构下`HIGHMEM`几乎用不到，此架构下内核空间的大小为`128T` 。在`x86`架构下超过内核空间`1g`的都在此区间
  2. `dma` 技术提供外设直接访问内存的方式，但是很多外设的寻址是16位的所以提供此区域方便外设访问
 
* `page` 物理页的管理对象也是`buddy system`的管理对象(中文很多时候都会有这种歧异>_<)。其在内存初始化是初始化在虚拟映射区的起点位置
* `numa node`和`zone` 关系  
  <img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/numa_node.png />
  
伙伴系统将内存的拆分成(2^n)个页的大小的快，相同大小的快用链表连接起来存放在`free_area`对象内。当前的 `amd64` 架构`n`的取值范围为 `0-10` 即最大单快的大小为 `2^10`个页。
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/buddy.png />

为了减少`cpu`之间对内存的竞争`zone`提供了 每个逻辑`core` 一个的块管理队列 `per_cpu_pageset`.当快使用完之后，符合要请求的快将会被释放到`per_cpu_pageset`中。
在次申请也会优先从`per_cpu_pageset`当前`core`的队列中获取块
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/zone_per_cpu.jpg  width="800" height="400" alt="zone中的per cpu的内存管理"/>

## 初始化
### numa的初始化
#### 调用链
```
/// amd64
setup_arch -> initmem_init -> x86_numa_init -> numa_init-> numa_register_memblks // 至此全局变量 struct pglist_data *node_data 已初始化完成
           -> x86_init.paging.pagetable_init -> sparse_init (稀疏内存初始化，此处不做阐述)
                                             -> zone_sizes_init -> zone_sizes_init -> free_area_init (Initialise all pg_data_t and zone data)
```
#### 生成物
* `numa_meminfo` 在`amd64`架构下`CONFIG_ACPI_NUMA`模式在`x86_acpi_numa_init`函数中初始化。`CONFIG_AMD_NUMA` 模式下`amd_numa_init`。本质是从启动信息中获取numa的配置信息
  `arm` 是从设备树，`x86_64`是从`pic`自举获得，不一而足。
  ```
  //// 为了方便理解才做如下定义，实际上他们的意思可能并非如此
  struct numa_memblk {
	  u64			start;  /// node 的地址起点
	  u64			end;   /// node的地址终点
	  int			nid;  /// node的id
  };
  struct numa_meminfo {
	  int			nr_blks;  /// node数量
	  struct numa_memblk	blk[NR_NODE_MEMBLKS]; 
  };
  static struct numa_meminfo numa_meminfo __initdata_or_meminfo;  /// 记录numa的配置信息
  ```
* `pglist_data *node_data` 最终生成物，此流程仅申请的物理空间。看起来并没有赋值
* 
 


## 接口
## 单块的分裂与合并
## 引用
https://blog.csdn.net/yhb1047818384/article/details/114454299
