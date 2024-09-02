## 说明
`buddy system` 可以说是内存核心，首先先要明确几个概念：
* `numa(Non Uniform Memory Access)` 是一种内存和`cpu`的关系设计的方案。其实现了每个`cpu(可以是物理cpu,也可以是多个core的集合)`有自己独立的内存群
  `numa` 中每个cpu都可以访问物理内存，但是访问自己的内存群的速度将会远远大于访问其他`cpu`的内存群。这在一定程度上也降低了`cpu`之间的内存资源竞争的消耗，即降低了同一内存
  不同cpu同时访问的概率
* `zone` 内核的物理内存是分区管理的，这是各逻辑概念仅是为了方便管理
  <img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/zone_type.webp />
  1. `amd64`架构下`HIGHMEM`几乎用不到，此架构下内核空间的大小为`128T` 。在`x86`架构下超过内核空间`1g`的都在此区间
  2. `dma` 技术提供外设直接访问内存的方式，但是很多外设的寻址是16位的所以提供此区域方便外设访问
 
* `page` 物理页的管理对象也是`buddy system`的管理对象(中文很多时候都会有这种歧异>_<)。其在内存初始化是初始化在虚拟映射区的起点位置
* `numa`和`zone` 关系  
  <img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/numa_node.png />
