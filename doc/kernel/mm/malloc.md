linux kernel 申请内存的方式有两种
即
 vmalloc 为代表的虚拟内存申请
 kmalloc 为代表的物理内存申请
vmalloc 本质上也是申请的kmalloc获取的物理内存


/// 调用链

vmalloc
***
                                                     -> __get_vm_area_node(获取虚拟内存相关)       
vmalloc -> __vmalloc_node -> __vmalloc_node_range -> 
                                          |          -> __vmalloc_area_node(申请物理内存) -> kmalloc_node
                                          |                          |(不满足条件会触发回调)
                                          -------------<-------------| 
***