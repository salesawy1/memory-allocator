# memory allocator

A first fit allocator over a single `mmap`'d arena, written to see what malloc is
actually doing underneath. `CustomAllocator<T>` implements the STL allocator
interface, so containers can be pointed at it:

```cpp
CustomAllocator<int> alloc(1 << 16);
std::vector<int, CustomAllocator<int>> v(alloc);
```

## Design

The arena is one anonymous mapping carved into blocks. Every block stores its size in
a header at the front and a footer at the back, and the low bit of that size doubles
as the allocated flag, which works because block sizes are rounded up to
`alignof(max_align_t)`. Free blocks are threaded onto a doubly linked list through
the payload space they are not using, so the list costs nothing while a block is out
on loan.

Allocation walks the list, takes the first block that fits, and splits it when the
leftover is still big enough to stand on its own. Freeing reads the boundary tags of
both physical neighbors and merges right away, so two adjacent free blocks never stay
separate.

Over aligned types get their payload pushed forward, which means the header is no
longer sitting directly behind the pointer the caller holds. The distance back to the
header is written into the word in front of the payload so `deallocate` can still
find it. With no padding that word is the header's own padding field, otherwise it
lands inside the gap.

A single mutex guards the arena. Copies and rebinds of the allocator share it through
a `shared_ptr`, which is what makes node based containers like `std::list` work.

## Running

```
./run.sh
```

That builds with ASan and UBSan and runs the tests: allocation and reuse, splitting,
coalescing, alignment including a 32 byte aligned type, exhaustion, `std::vector` and
`std::list`, and a randomized alloc and free loop that pattern fills every block and
checks the arena collapses back to one free block once nothing is live.

## Limitations

The arena is a fixed size picked at construction, 4096 bytes by default, and it never
grows. `allocate` throws `bad_alloc` once it is full.

Every default constructed allocator maps its own arena, so a container that default
constructs one pays for a full region.

One free list and first fit, no size classes or bins. Coalescing is the only thing
working against fragmentation.

One lock for the whole arena. Thread safe, but not something you would put under
contention.

`deallocate` has to be `noexcept`, so a pointer that did not come from the arena is
ignored instead of reported.
