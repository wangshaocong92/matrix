## slab 技术
slab 是为了降低内存浪费而引入的技术
## 调用链
### 初始化调用链

### new slab 调用链
<img src=https://github.com/wangshaocong92/matrix/blob/main/doc/image/new_slab.png />


## 资源
* `struct slab` 

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
					void *freelist;		/// slab可以看成是一个数组，此指针指向下一个未使用的object的地址
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