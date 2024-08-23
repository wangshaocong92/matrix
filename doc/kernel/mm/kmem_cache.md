## slab 技术
slab 是为了降低内存浪费而引入的技术
## 调用链
### 初始化调用链
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/kmem_init.png />

### new slab 调用链
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/new_slab.png />

## 资源
* `struct slab`  slab 是此技术的核心，也是此技术的最小单元，内存池管理是其主要工作
```
struct slab {
	unsigned long __page_flags;   //// 当前 page的flag

	struct kmem_cache *slab_cache;  //// slab 所属的 kmem_cache
	union {
		struct {
			union {
				struct list_head slab_list; //// slab 链表，其头item在kmem_cache_node的partial 保存
#ifdef CONFIG_SLUB_CPU_PARTIAL
				struct {
					struct slab *next;
					int slabs;	/* Nr of slabs left */
				};
#endif
			};
			/* Double-word boundary */
			union {
				struct {
					void *freelist;		/// 当前 slab的free 链表的header指针
					union {
						unsigned long counters;
						struct {
							unsigned inuse:16;    /// 已经在使用的object的数量
							unsigned objects:15;  /// 当前slab obejct的总量
							unsigned frozen:1;
						};
					};
				};
#ifdef system_has_freelist_aba
				freelist_aba_t freelist_counter;
#endif
			};
		};
		struct rcu_head rcu_head;
	};
	unsigned int __unused;

	atomic_t __page_refcount;
#ifdef CONFIG_MEMCG
	unsigned long memcg_data;
#endif
};
```
* `struct kmem_cache_node` kmem_cache 的单node 扩充
```
struct kmem_cache_node {
	spinlock_t list_lock;  // 锁
	unsigned long nr_partial; // partial 的数量
	struct list_head partial; /// slab 链表
};
```
* `struct kmem_cache` slab算法的管理单元，slab的申请主要通过此对象的参数来处理.linux 看起来没有统一管理的概念哪个结构体需要被其管理就可以申请一个
```
#ifndef CONFIG_SLUB_TINY
	struct kmem_cache_cpu __percpu *cpu_slab;
#endif
	/* Used for retrieving partial slabs, etc. */
	slab_flags_t flags;                      //// 以每一位的0和1来确认当前slab内存分配的细节 具体见 `enum _slab_flag_bits`
	unsigned long min_partial;               //// min_partial 参数指定了每个 SLAB 缓存（kmem_cache）中最少应该保留的部分使用的 slab 的数量。 通过调整 min_partial，内核可以在内存占用和
                                             //// 分配 效率之间找到平衡，从而优化系统的内存管理和性能表现。
 	unsigned int size;		/* Object size including metadata */
	unsigned int object_size;	/* Object size without metadata */
	struct reciprocal_value reciprocal_size;        //// 用来优化除法的参数，主要用来使用乘法和位移等操作来降低cpu消耗
	unsigned int offset;		/* Free pointer offset */  //// 一种将指针放到数据中间的技巧，降低越界的风险。因为slab越界写是不会报错的，若是越界将保存的地址信息乱写，很可能下次访问会
                                                           //// 访问到不可预知的区域。放到中间更安全一些
                                                           //// set_freepointer 和 get_freepointer 以及 freeptr_t 完成了此工作
                                                           //// 此偏移指向的地址存储在 object中，只在object free的时候起效、
                                                           //// 在某些情况下可能在对齐object之后
                                                           //// 这些在下文阐述
#ifdef CONFIG_SLUB_CPU_PARTIAL
	/* Number of per cpu partial objects to keep around */
	unsigned int cpu_partial; 
	/* Number of per cpu partial slabs to keep around */
	unsigned int cpu_partial_slabs;
#endif
	struct kmem_cache_order_objects oo; //// 前16位内的是单个slab object的个数。之后的是单个slab 占有的page的个数

	/* Allocation and freeing of slabs */
	struct kmem_cache_order_objects min;
	gfp_t allocflags;		/* gfp flags to use on each alloc */ //// 内存申请的区域
	int refcount;			/* Refcount for slab cache destroy */ //// 引用记数
	void (*ctor)(void *object);	/* Object constructor */
	unsigned int inuse;		/* Offset to metadata */ //// object size 对齐后的结果，以此可以用来判断 下一个free item的指针是在object的中部还是尾部
	unsigned int align;		/* Alignment */          ////  内核会综合 word size , cache line ，align 计算出一个合理的对齐尺寸
	unsigned int red_left_pad;	/* Left redzone padding size */  /// 调试使用
	const char *name;		/* Name (only for display!) */   ///// 无用
	struct list_head list;		/* List of slab caches */ //// 记录所有的 slab
#ifdef CONFIG_SYSFS
	struct kobject kobj;		/* For sysfs */
#endif
#ifdef CONFIG_SLAB_FREELIST_HARDENED
	unsigned long random;
#endif

#ifdef CONFIG_NUMA
	/*
	 * Defragmentation by allocating from a remote node.
	 */
	unsigned int remote_node_defrag_ratio;
#endif

#ifdef CONFIG_SLAB_FREELIST_RANDOM
	unsigned int *random_seq;
#endif

#ifdef CONFIG_KASAN_GENERIC
	struct kasan_cache kasan_info; //// 控制在内存分配时是否允许从远程节点（相对于本地节点）进行碎片整理（defragmentation）。
#endif

#ifdef CONFIG_HARDENED_USERCOPY
	unsigned int useroffset;	/* Usercopy region offset */    /// ok 先不管他
	unsigned int usersize;		/* Usercopy region size */ 
#endif

	struct kmem_cache_node *node[MAX_NUMNODES]; //// 单个 kmem_cache是跨cpu的，所以需要一个对每个cpu进行管理的node 数组
```
## slab 内存布局
### kmem_cache 布局
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/kmem_cache.jpg />

### slab 布局
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/slab.jpg />

### slab free item
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/slab_item.jpg />

### slab busy item
...

## 操作
### new object 
### free object
`slab_free`
```
                                                                       -> call_rcu(SLAB_TYPESAFE_BY_RCU)
slab_free -> do_slab_free -> __slab_free(slab_empty) -> discard_slab ->  
                                                                       -> __free_slab
```
#### virt_to_slab
从object的虚拟地址找到相应的slab对象
```
    void * object;
    /// 从当前虚拟地址，映射到page 变量的地址
    page *page = virt_to_page(object); => pfn_to_page(__phys_addr(object) >> PAGE_SHIFT = pfn) => (vmemmap + (pfn)); 
    folio * folio = page_folio(page); => _compound_head(page) => page->compound_head - 1(page 不是fake head page) 
                                                              => page_fixed_fake_head
```
##### `page` 和物理地址的映射关系
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/page_to_ph.png />

* page的数组是保存在vmolloc区的首部，且为虚拟地址存储
* page的数组的排序和内存物理地址是一致的，一个page就是物理地址同位置的页
##### 复合`page`寻址
* 复合`page`中，`head page`保存大多数块的数据,`tail pages` 存储各自的数据。同时`tail pages` 的 `compound_head` 参数中存储着`head page`的地址
* `compound_head` 首个bit位置一来确认其有效可用，所以`compound_head = (ulong)(page *) + 1`,因 `sizeof(page)` 是偶数 正常其地址的右起首个bit不可能位1

#### free_slab
* `SLAB_TYPESAFE_BY_RCU` rcu 则 走 `call_rcu`,延迟释放
* 立刻释放走 `__free_slab`

#### __free_slab
* `__slab_clear_pfmemalloc` page `active` flag 清空
* `mm_account_reclaimed_pages`  统计和记录在内存回收过程中回收的页面数量
* `unaccount_slab` 标记一下slab的释放 
* `__free_pages` 释放pages




