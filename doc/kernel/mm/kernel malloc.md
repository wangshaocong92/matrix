
## 内核虚拟内存
### 布局
```
 64位 Linux 虚拟内存布局 (x86_64) l4寻址(48bit)
  ----------------------------------------
  | 用户空间       |    0x0000000000000000  -  0x00007fffffffffff
  ----------------------------------------
  | 未使用区       |    0x0000800000000000  -  0xffff7fffffffffff
  ----------------------------------------
  | 直接映射区     |    0xffff888000000000  -  0xffffc87fffffffff
  ----------------------------------------
  | vmalloc 区域   |    0xffffc88000000000  -  0xffffe8ffffffffff
  ----------------------------------------
  | 模块加载区     |    0xffffffffa0000000  -  0xffffffffc0000000
  ----------------------------------------
  | 内核映像       |    0xffffffff80000000  -  0xffffffffffc00000
  ----------------------------------------
  | 固定映射区     |    0xffffffffff000000  -  0xffffffffffffffff
  ----------------------------------------

    32位 Linux 虚拟内存布局 (x86)
  ----------------------------------------
  | 用户空间       |    0x00000000  -  0xbfffffff
  ----------------------------------------
  | 高内存映射区   |    0xc0000000  -  0xdfffffff
  ----------------------------------------
  | `vmalloc` 区域 |    0xe0000000  -  0xefffffff
  ----------------------------------------
  | 固定映射区     |    0xf0000000  -  0xf7ffffff
  ----------------------------------------
  | 模块加载区     |    0xf8000000  -  0xfbffffff
  ----------------------------------------
  | 内核映像       |    0xfc000000  -  0xffffffff
  ----------------------------------------
```
### 内存申请
linux kernel 申请内存的方式有两种
* vmalloc 为代表的内存内核区虚拟内存申请，在 VMALLOC_START - VMALLOC_END(64xl4页表: 0xffffc90000000000UL - 0xffffe8ffffffffffUL) 之间
* kmalloc 为代表的固定偏移物理内存的申请，其申请区域在内存内核区的直接映射区

#### 注意：
* 内核内存申请并不会真的获取到真实的物理地址，以上两种方式获取到的都是虚拟内存地址
* 内存申请不会真的去物理内存上面获取一块内存，而是从类似于伙伴系统中获取到合适的物理内存
* vmalloc 和kmalloc除了分配的地址区不同之外最大的区别在于vmalloc 获取到的大内存块(超过一个page)在真是物理内存上并不一定是连续的。而kmalloc获取到的内存块确定是连续的内存。所以内存块访问来说k相对于v来说有两大优势
  1. k不需要多级寻址，虚到实速度更快
  2. 大内存访问k不需要多次寻址，单次寻址既可持续访问


## vmalloc
### 调用链

```
                                                     -> __get_vm_area_node(获取 虚拟内存相关)       
vmalloc -> __vmalloc_node -> __vmalloc_node_range -> 
                                                     -> __vmalloc_area_node(申  请物理内存) -> vm_area_alloc_pages(申请物理内存)
                                                    
```
### vmalloc 虚拟内存的管理组件
* `struct vmap_area` 顾名思义此结构体描述了一块虚拟内存。虚拟内存管理的重要结构体，其是多种虚拟内存管理红黑树的节点.
```
    struct vmap_area {
	    unsigned long va_start;   //// 虚拟内存的起点地址
	    unsigned long va_end;     //// 虚拟内存的终点地址
    
	    struct rb_node rb_node;   //// 红黑树的节点，使用 rb_entry 可以从此变量的地址找到vmap_area的地址
	    struct list_head list;    //// 链表的节点，同上
    
	    union {
	    	unsigned long subtree_max_size;  //// 在 free tree 中使用，
                                             //// 其描述了当前节点以及当前节点子孙后代节点中的最大容量即 va_end - va_start
	    	struct vm_struct *vm;            //// 在 busy tree 中使用其描述了虚拟内存 和 物理内存的关系
	    };
	    unsigned long flags;                //// 是否启用 vm_map_ram，即虚拟内存映射到整块物理内存上？
    };

```
* `struct vm_struct` 是 `vmap_area` busy模式下的扩展，管理虚拟内存和物理内存的关系
```
    struct vm_struct {
    	struct vm_struct	*next;  //// 在vmalloc初始化之前做单向链表使用
    	void			*addr;      //// 虚拟内存地址
    	unsigned long		size;   //// 虚拟内存块大小
    	unsigned long		flags;  
    	struct page		**pages;    //// 1. 指向一个直接内存区内存地址(并非实际物理地址，会存在偏移)
                                    //// 2. 指向一个虚拟内存地址  
                                    //// 3. 此内存地址中保存着一个 nr_pages 大小的page数组
    #ifdef CONFIG_HAVE_ARCH_HUGE_VMALLOC
    	unsigned int		page_order;
    #endif
    	unsigned int		nr_pages;   //// 页面大小
    	phys_addr_t		phys_addr;    //// ioremap操作会将设备的物理地址赋值给此变量，应该是一个快速访问的工具吧，毕竟虚拟地址也可以转成物理地址
    	const void		*caller;
    };
```
* `free_vmap_area_root` 虚拟内存 free tree，全局唯一
* `vmap_nodes` 所有node的vmap,此变量和cpu的个数有关在 `vmap_init_nodes` 做正式初始化，busy tree 受此管理
* `vmlist` 在vmalloc初始化之前的boot期内核也需要一些虚拟内存，故提供此链表来记录.
  1. 对象存储在内存的`.init.data` 区，boot结束后释放
  2. 内核使用 `vm_area_register_early`和 ``vm_area_add_early` 函数供外部使用
```
    void __init vm_area_register_early(struct vm_struct *vm, size_t align)
    {
	    unsigned long addr = ALIGN(VMALLOC_START, align);
	    struct vm_struct *cur, **p;

	    BUG_ON(vmap_initialized);

	    for (p = &vmlist; (cur = *p) != NULL; p = &cur->next) {
	    	if ((unsigned long)cur->addr - addr >= vm->size)
	    		break;
	    	addr = ALIGN((unsigned long)cur->addr + cur->size, align);
	    }

	    BUG_ON(addr > VMALLOC_END - vm->size);
	    vm->addr = (void *)addr;
	    vm->next = *p;
	    *p = vm;
	    kasan_populate_early_vm_area_shadow(vm->addr, vm->size);
    }
```
* `struct vmap_block_queue` 管理 `vmap_block` 对象的队列,`vmolloc` 中每个cpu存在一个此对象。可使用`raw_cpu_ptr` 取用
  ```
  struct vmap_block_queue {
	  spinlock_t lock;
	  struct list_head free; /// list_head 设计的真不错，可以接收任何存在此变量的结构体做链表，当前变量为 vmap_block 队列的头指针

	  /*
	   * An xarray requires an extra memory dynamically to
	   * be allocated. If it is an issue, we can use rb-tree
	   * instead.
	   */
	  struct xarray vmap_blocks;
  };
  ```
* 
### 初始化
#### 调用链
```
                                             -> KMEM_CACHE
start_kernel -> mm_core_init -> vmalloc_init -> vmap_init_nodes(vmap_nodes 初始化)
                                             -> vmap_init_free_space(free space 初始化)
```
#### 代码
```
  ...
	vmap_area_cachep = KMEM_CACHE(vmap_area, SLAB_PANIC);
  //// vmap_block_queue 暂时不做阐述
	for_each_possible_cpu(i) {
		struct vmap_block_queue *vbq;
		struct vfree_deferred *p;

		vbq = &per_cpu(vmap_block_queue, i);
		spin_lock_init(&vbq->lock);
		INIT_LIST_HEAD(&vbq->free);
		p = &per_cpu(vfree_deferred, i);
		init_llist_head(&p->list);
		INIT_WORK(&p->wq, delayed_vfree_work);
		xa_init(&vbq->vmap_blocks);
	}

	/*
	 * Setup nodes before importing vmlist. 因为 busy tree 初始化需要再 vmlist展开之前
	 */
	vmap_init_nodes();

	/* Import existing vmlist entries. vmlist 内存储的是 busy va,需要将其加入到 busy tree中*/
	for (tmp = vmlist; tmp; tmp = tmp->next) {
		va = kmem_cache_zalloc(vmap_area_cachep, GFP_NOWAIT);
		if (WARN_ON_ONCE(!va))
			continue;

		va->va_start = (unsigned long)tmp->addr;
		va->va_end = va->va_start + tmp->size;
		va->vm = tmp;

		vn = addr_to_node(va->va_start);
		insert_vmap_area(va, &vn->busy.root, &vn->busy.head);
	}

	/*
	 * Now we can initialize a free vmap space. 
   * 因为虚拟内存是 0x0~0xf*16的完整内存块，所以知道了busy va。自然free va也就知道了，本函数根据busy tree填充 free tree和list 
	 */
	vmap_init_free_space();
  ...
```

### free tree 维护
* free_vmap_area_root 初始化，`vmap_init_free_space`
```
  unsigned long vmap_start = 1;
	const unsigned long vmap_end = ULONG_MAX;
	struct vmap_area *free;
	struct vm_struct *busy;

	/*
	 *     B     F     B     B     B     F
	 * -|-----|.....|-----|-----|-----|.....|-
	 *  |           The KVA space           |
	 *  |<--------------------------------->|
	 */
	for (busy = vmlist; busy; busy = busy->next) {
		/*
		 因为在虚拟内存空间free和busy node都是在同一块连续的地址为 0x0-0xffffffffffffffff 大内存之内的。如上图的注释
		 所以只要是将说有busy遍历过了，free自然也就遍历过了
		*/
		if ((unsigned long) busy->addr - vmap_start > 0) {
			free = kmem_cache_zalloc(vmap_area_cachep, GFP_NOWAIT);
			if (!WARN_ON_ONCE(!free)) {
				free->va_start = vmap_start;
				free->va_end = (unsigned long) busy->addr;

				insert_vmap_area_augment(free, NULL,
					&free_vmap_area_root,
						&free_vmap_area_list);
			}
		}

		vmap_start = (unsigned long) busy->addr + busy->size;
	}
	//// 前面只处理了中间的，此处处理了最后一段
	if (vmap_end - vmap_start > 0) {
		free = kmem_cache_zalloc(vmap_area_cachep, GFP_NOWAIT);
		if (!WARN_ON_ONCE(!free)) {
			free->va_start = vmap_start;
			free->va_end = vmap_end;

			insert_vmap_area_augment(free, NULL,
				&free_vmap_area_root,
					&free_vmap_area_list);
		}
	}
```
* insert free area to tree and list `insert_vmap_area_augment`  `insert_vmap_area` 
* 根据申请的内存地址大小，重新处理 free tree  `va_clip`
    1. re free tree 并不会直接取走当前的area，而是在busy tree中插入新建的area所以当前的area需要重新处理
    2. 如代码所示申请的内存存在于当前area的不同位置将会出现不同的处理方式
  ```
  struct vmap_area *lva = NULL;
  enum fit_type type = classify_va_fit_type(va, nva_start_addr, size);  
  if (type == FL_FIT_TYPE) {
  	/*
  	 * No need to split VA, it fully fits. 直接删掉
  	 *
  	 * |               |
  	 * V      NVA      V
  	 * |---------------|
  	 */
  	unlink_va_augment(va, root);
  	kmem_cache_free(vmap_area_cachep, va);
  } else if (type == LE_FIT_TYPE) {
  	/*
  	 * Split left edge of fit VA.
  	 *
  	 * |       |
  	 * V  NVA  V   R
  	 * |-------|-------|
  	 */
  	va->va_start += size;
  } else if (type == RE_FIT_TYPE) {
  	/*
  	 * Split right edge of fit VA.
  	 *
  	 *         |       |
  	 *     L   V  NVA  V
  	 * |-------|-------|
  	 */
  	va->va_end = nva_start_addr;
  } else if (type == NE_FIT_TYPE) {
  	/*
  	 * Split no edge of fit VA. 需要添加一个新的area
  	 *
  	 *     |       |
  	 *   L V  NVA  V R
  	 * |---|-------|---|
  	 */
  	 //// 既然左右都有则在左侧新增一个
  	lva = __this_cpu_xchg(ne_fit_preload_node, NULL); 
  	/*
  	 * Build the remainder.
  	 */
  	lva->va_start = va->va_start;
  	lva->va_end = nva_start_addr; 
  	/*
  	 * Shrink this VA to remaining size.
  	 */
  	va->va_start = nva_start_addr + size;
  } else {
  	return -1;
  } 
  if (type != FL_FIT_TYPE) {
  	augment_tree_propagate_from(va);
  	//// 将新添加的va 添加到红黑树中
  	if (lva)	/* type == NE_FIT_TYPE */
  		insert_vmap_area_augment(lva, &va->rb_node, root, head);
  } 
  ```

### 虚拟内存分配 
```
__get_vm_area_node -> alloc_vmap_area -> __alloc_vmap_area -> find_vmap_lowest_match
```
* __get_vm_area_node
  1. 构造 `vm_struct *area`
  2. 在`free tree`中找到和申请内存大小匹配的 `vmap_area`,并新建新的`va`将其放到 `busy tree` 中
  3. setup `area`
```
  static struct vm_struct *__get_vm_area_node(unsigned long size,
  		unsigned long align, unsigned long shift, unsigned long flags,
  		unsigned long start, unsigned long end, int node,
  		gfp_t gfp_mask, const void *caller)
  {
      ...
      area = kzalloc_node(sizeof(*area), gfp_mask & GFP_RECLAIM_MASK, node);
  	  ...
  	  va = alloc_vmap_area(size, align, start, end, node, gfp_mask, 0);
  	  ...
  	  setup_vmalloc_vm(area, va, flags, caller);
      ... 
  	  return area;
  }
```
* find_vmap_lowest_match
  1. 虚拟内存查找说起来就一句话，在free tree中找到一块容量大于需求内存`size`的地址最低的 `va`
  2. 此处主要使用了subtree_max_size 此成员记录了当前节点和其子孙节点的容量的最大值

### 物理内存分配
```
__vmalloc_area_node-> vm_area_alloc_pages 
                   -> vmap_pages_range
```
* __vmalloc_area_node
  1. area->pages 初始化,若是一个页可以放下pages数组则直接申请一块物理内存，否则申请一块虚拟内存来存储它即回调`__vmalloc_node`
```
        unsigned long size = get_vm_area_size(area);  //// 申请内存块的大小
        unsigned int nr_small_pages = size >> PAGE_SHIFT; //// 内存块需要占用多少个页 ？
        array_size = (unsigned long)nr_small_pages * sizeof(struct page *); /// 页的个数 * 单个页地址的大小
        if (array_size > PAGE_SIZE) {
	    	//// area->pages 赋值为了新申请的虚拟内存的起始地址
	    	area->pages = __vmalloc_node(array_size, 1, nested_gfp, node,
	    				area->caller);
	    } else {
	    	area->pages = kmalloc_node(array_size, nested_gfp, node);
	    }
```
  2. page order 设置即 set_vm_area_page_order 大连续页申请，看起来是没有实现的
```
在内核中，页顺序（page order）表示要分配的物理内存块的大小。order 为 0 表示一个单独的页面（通常是 4 KB），order 为 1 表示连续的两个页面，order 为 2 表示连续的四个页面，以此类推。通过设置页顺序，内核可以分配更大的连续内存块，这在需要减少页表项开销或提高内存访问效率的场景中尤为重要。

    if (!order) {
		...
	} else if (gfp & __GFP_NOFAIL) {
		/*
		 * Higher order nofail allocations are really expensive and
		 * potentially dangerous (pre-mature OOM, disruptive reclaim
		 * and compaction etc.
		 */
		alloc_gfp &= ~__GFP_NOFAIL;
		nofail = true;
	}

```
  3. `vm_area_alloc_pages` 申请物理内存,从cpu管理的 `per_cpu_pageset` 列表中找到足够的`page`赋值到`vm_struct`的`page`中
```
    ...
    pcp = pcp_spin_trylock(zone->per_cpu_pageset);
    pcp_list = &pcp->lists[order_to_pindex(ac.migratetype, 0)];
    while (nr_populated < nr_pages) { 
    	page = __rmqueue_pcplist(zone, 0, ac.migratetype, alloc_flags,
    							pcp, pcp_list);
      ...
    	nr_account++; 
    	prep_new_page(page, 0, gfp, 0);
    	if (page_list)
    		list_add(&page->lru, page_list);
    	else
    		page_array[nr_populated] = page;
    	nr_populated++;
    }
```
  4. `vmap_pages_range` 将虚拟内存和物理内存的映射写入TLB
## IOREMAP 技术
IOREMAP 提供了一系列物理内存地址转虚拟内存地址的接口。因linux系统在有mmu的情况下不允许直接访问物理内存，所以外加设备的内存和寄存器需要映射到虚拟内存上，这样就可以正常访问了
### 接口
* ioremap  _PAGE_CACHE_MODE_UC_MINUS mode下的内存映射
* ioremap_uc  _PAGE_CACHE_MODE_UC mode下的内存映射
* ioremap_wc _PAGE_CACHE_MODE_WC mode下的内存映射
* ioremap_wt _PAGE_CACHE_MODE_WT mode下的内存映射
* ioremap_encrypted _PAGE_CACHE_MODE_WB mode下，加密的的内存映射，
* ioremap_prot 自定义缓存模式的内存映射
* ioremap_cache _PAGE_CACHE_MODE_WB mode下的内存映射
* iounmap 取消内存映射此接口需要和以上接口成对出现
  ```
  enum page_cache_mode {
	  _PAGE_CACHE_MODE_WB       = 0, /// Write-Back 写回缓存模式是指数据在写入缓存时不会立即写入主存，而是在缓存中的数据被驱逐（替换）时才写回主存。
	  _PAGE_CACHE_MODE_WC       = 1, /// Write-Combining 合并多个写操作后，再进行一次性写入，以减少写操作的次数。
	  _PAGE_CACHE_MODE_UC_MINUS = 2, /// Uncached Minus 内存页面是非缓存的，且通常比普通的 Uncached (UC) 模式更为严格。这意味着对这些页面的访问不会利用 CPU 缓存，而且可能会在硬件层面禁止某些优化，如合并  写入操作。
	  _PAGE_CACHE_MODE_UC       = 3, /// Uncached 数据直接从磁盘读取或写入磁盘，不经过缓存。
	  _PAGE_CACHE_MODE_WT       = 4, /// Write-Through  每次写操作都会同时写入内存和磁盘，确保数据的一致性。
	  _PAGE_CACHE_MODE_WP       = 5, /// Write-Protect 在写保护模式下，内存页面是只读的，任何对该页面的写操作都会触发页面错误（Page Fault）。这种模式通常用于保护内存中的关键数据，防止意外或恶意的写操作

	  _PAGE_CACHE_MODE_NUM      = 8
  };
  ```
### 实现
#### 调用流程
```                         
                            -> memtype_reserve(存储当前地址的pcm到memtype_rbroot中)
ioremap -> __ioremap_caller -> get_vm_area_caller(获取一个 va 并将其加入到busy tree)
                            -> ioremap_page_range -> vmap_page_range (将虚实映射写入到TLB)
```

#### 页对齐
```
	offset = phys_addr & ~PAGE_MASK;
	phys_addr &= PAGE_MASK;
	size = PAGE_ALIGN(last_addr+1) - phys_addr;
```






  



