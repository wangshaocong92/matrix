
```
                                ----                                                                                                  ...                   -----------|  
                                |                                                                                        -> early_kmem_cache_node_alloc (初始化应该走此分支)->|
mm_core_init -> kmem_cache_init |-> create_boot_cache -> __kmem_cache_create -> kmem_cache_open -> init_kmem_cache_nodes -> kmem_cache_alloc_node -> slab_alloc_node
-> __slab_alloc_node -------------> new_slab -> allocate_slab -> alloc_pages_node -> __alloc_pages_node -> __alloc_pages -> get_page_from_freelist
            |        -> get_partial -> get_partial_node (从已注册的kmem_cache_node中获取到一个slab，且已经在slab上面申请到了内 存)
            |
            |------> alloc_single_from_new_slab(new_slab 之后的处理) -> add_partial(将申请到的slab 加到对应的 kmem_cache_node的链表中)

```