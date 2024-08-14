
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
    	struct page		**pages;    //// 1. 指向一个物理页面内存地址(并非实际物理地址，会存在偏移)
                                    //// 2. 指向一个虚拟内存地址  
                                    //// 3. 此内存地址中保存着一个 nr_pages 大小的page数组
    #ifdef CONFIG_HAVE_ARCH_HUGE_VMALLOC
    	unsigned int		page_order;
    #endif
    	unsigned int		nr_pages;   //// 页面大小
    	phys_addr_t		phys_addr;
    	const void		*caller;
    };
```
* `static struct rb_root free_vmap_area_root = RB_ROOT` 虚拟内存 free tree，全局唯一
* `static struct vmap_node *vmap_nodes = &single` 所有node的vmap,此变量和cpu的个数有关在 `vmap_init_nodes` 做正式初始化，busy tree 受此管理
* `static struct vm_struct *vmlist __initdata` 在vmalloc初始化之前的boot期内核也需要一些虚拟内存，故提供此链表来记录.
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
### 目标虚拟内存块分配 
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

### free tree 维护
#### free_vmap_area_root 初始化


## 物理内存分配
```
__vmalloc_area_node-> vm_area_alloc_pages
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
  3. vm_area_alloc_pages 申请物理内存


  



