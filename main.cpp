#include <iostream>
#include <vector>
#include <list>
#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <random>
#include <cassert>
#include <utility>
#include <new>
#include <mutex>
#include <memory>
#include <type_traits>

// memory allocator
// customallocator<t> is an stl compatible allocator sitting on one mmap'd arena
// boundary tags, explicit free list, first fit with splitting, coalescing on free
// the arena is a fixed size (4096 bytes unless you ask for more) and every copy or
// rebind of the allocator shares it, allocations that don't fit throw bad_alloc

constexpr size_t ALIGNMENT = alignof(std::max_align_t);

constexpr size_t align_up(size_t n) {
    return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

struct BlockFooter {
    size_t block_size;

    // block size is a multiple of 2 so lsb is free for allocated flag
    BlockFooter(size_t block_size, bool allocated)
        : block_size((block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated)) {}

    bool is_allocated() const {
        return block_size & 1;
    }
    size_t get_block_size() const {
        return block_size & ~static_cast<size_t>(1);
    }
    void resize(size_t new_size) {
        // keep allocation flag
        block_size = new_size | (block_size & 1);
    }
    void change_allocation(bool allocated) {
        block_size = (block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated);
    }
};

struct BlockHeader {
    size_t block_size;
    size_t padding;

    // block size is a multiple of 2 so lsb is free for allocated flag
    BlockHeader(size_t block_size, bool allocated)
        : block_size((block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated)), padding(0) {}

    bool is_allocated() const {
        return block_size & 1;
    }
    size_t get_block_size() const {
        return block_size & ~static_cast<size_t>(1);
    }
    void resize(size_t new_size) {
        block_size = (block_size & 1) | new_size;
    }
    void change_allocation(bool allocated) {
        block_size = (block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated);
    }
    unsigned char* get_payload_ptr() const {
        return reinterpret_cast<unsigned char*>(const_cast<BlockHeader*>(this)) + sizeof(BlockHeader) + padding;
    }
};

struct FreeBlockHeader {
    size_t block_size;
    struct {
        FreeBlockHeader* next;
        FreeBlockHeader* prev;
    } ptrs;

    // block size is a multiple of 2 so lsb is free for allocated flag
    FreeBlockHeader(size_t block_size, bool allocated)
        : block_size((block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated))
    {
        ptrs.next = nullptr;
        ptrs.prev = nullptr;
    }

    bool is_allocated() const {
        return block_size & 1;
    }
    size_t get_block_size() const {
        return block_size & ~static_cast<size_t>(1);
    }
    void resize(size_t new_size) {
        block_size = (block_size & 1) | new_size;
    }
    void change_allocation(bool allocated) {
        block_size = (block_size & ~static_cast<size_t>(1)) | static_cast<size_t>(allocated);
    }
};

constexpr size_t METADATA_SIZE = sizeof(FreeBlockHeader) + sizeof(BlockFooter);
constexpr size_t METADATA_SIZE_ALLOC = sizeof(BlockHeader) + sizeof(BlockFooter);
constexpr size_t INITIAL_ALLOCATOR_SIZE = 4096;
constexpr size_t MINIMUM_PAYLOAD_SIZE = 4;
constexpr size_t MINIMUM_BLOCK_SIZE = align_up(METADATA_SIZE + MINIMUM_PAYLOAD_SIZE);

// block starts and header sizes are both multiples of ALIGNMENT, so any padding a
// payload needs is one too and is either zero or wide enough to hold a size_t
static_assert(sizeof(BlockHeader) % ALIGNMENT == 0);
static_assert(ALIGNMENT >= sizeof(size_t));

class Arena {
public:
    explicit Arena(size_t size = INITIAL_ALLOCATOR_SIZE)
        : data_(nullptr), head_(nullptr), size_(align_up(size < MINIMUM_BLOCK_SIZE ? MINIMUM_BLOCK_SIZE : size))
    {
        void* region = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(region == MAP_FAILED)
            throw std::bad_alloc();
        data_ = static_cast<unsigned char*>(region);
        // one free block covering the whole region
        head_ = new(data_) FreeBlockHeader(size_, false);
        new(get_footer(data_, size_)) BlockFooter(size_, false);
    }
    ~Arena() {
        if(data_)
            munmap(data_, size_);
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* allocate(size_t bytes, size_t alignment) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(alignment < ALIGNMENT)
            alignment = ALIGNMENT;
        if(bytes > size_ - METADATA_SIZE_ALLOC)
            throw std::bad_alloc();

        size_t padding_needed = 0;
        size_t needed = 0;
        FreeBlockHeader* curr_block = head_;
        while(curr_block) {
            unsigned char* start = reinterpret_cast<unsigned char*>(curr_block);
            size_t padding = get_alignment_offset(start + sizeof(BlockHeader), alignment);
            size_t total = align_up(METADATA_SIZE_ALLOC + padding + bytes);
            if(total <= curr_block->get_block_size()) {
                padding_needed = padding;
                needed = total;
                break;
            }
            curr_block = curr_block->ptrs.next;
        }
        if(!curr_block)
            throw std::bad_alloc();

        // split the free block if what's left over can still hold a block of its own
        size_t original_size = curr_block->get_block_size();
        if(original_size - needed >= MINIMUM_BLOCK_SIZE)
            split_block(curr_block, needed, original_size - needed);
        else {
            remove_from_free_list(curr_block);
            needed = original_size;
        }

        unsigned char* start = reinterpret_cast<unsigned char*>(curr_block);
        BlockHeader* block = new(start) BlockHeader(needed, true);
        block->padding = padding_needed;
        new(get_footer(start, needed)) BlockFooter(needed, true);

        // an over aligned t pushes the payload past the header, so leave the distance
        // back to the header in the word right in front of the payload. with no padding
        // that word is the header's own padding field, otherwise it lands in the gap
        unsigned char* payload = block->get_payload_ptr();
        *reinterpret_cast<size_t*>(payload - sizeof(size_t)) = padding_needed;
        return payload;
    }

    void deallocate(void* p) noexcept {
        if(!p)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        unsigned char* ptr = static_cast<unsigned char*>(p);
        if(!within_boundary(ptr))
            return;

        BlockHeader* block = get_block_header(ptr);
        size_t block_size = block->get_block_size();
        unsigned char* start = reinterpret_cast<unsigned char*>(block);
        FreeBlockHeader* free_block = new(start) FreeBlockHeader(block_size, false);
        get_footer(start, block_size)->change_allocation(false);
        perform_coalescence(free_block);
    }

private:
    std::mutex mutex_;
    unsigned char* data_;
    FreeBlockHeader* head_;
    size_t size_;

    void remove_from_free_list(FreeBlockHeader* block) noexcept {
        if(block == head_)
            head_ = block->ptrs.next;
        if(block->ptrs.prev)
            block->ptrs.prev->ptrs.next = block->ptrs.next;
        if(block->ptrs.next)
            block->ptrs.next->ptrs.prev = block->ptrs.prev;
        block->ptrs.next = nullptr;
        block->ptrs.prev = nullptr;
    }

    void push_front(FreeBlockHeader* block) noexcept {
        block->ptrs.prev = nullptr;
        block->ptrs.next = head_;
        if(head_)
            head_->ptrs.prev = block;
        head_ = block;
    }

    inline bool within_boundary(unsigned char* ptr) const noexcept {
        return (ptr >= data_) && (ptr < data_ + size_);
    }

    inline BlockFooter* get_footer(unsigned char* start, size_t block_size) const noexcept {
        return reinterpret_cast<BlockFooter*>(start + block_size - sizeof(BlockFooter));
    }

    inline BlockHeader* get_block_header(unsigned char* payload) const noexcept {
        size_t padding = *reinterpret_cast<size_t*>(payload - sizeof(size_t));
        return reinterpret_cast<BlockHeader*>(payload - sizeof(BlockHeader) - padding);
    }

    // the block in front is only reachable through its footer, the one behind starts
    // with a header either way so its flag can be read directly
    inline FreeBlockHeader* get_prev_free_block(FreeBlockHeader* block) const noexcept {
        unsigned char* start = reinterpret_cast<unsigned char*>(block);
        if(start == data_)
            return nullptr;
        BlockFooter* prev_footer = reinterpret_cast<BlockFooter*>(start - sizeof(BlockFooter));
        if(prev_footer->is_allocated())
            return nullptr;
        unsigned char* prev_start = start - prev_footer->get_block_size();
        if(!within_boundary(prev_start))
            return nullptr;
        return reinterpret_cast<FreeBlockHeader*>(prev_start);
    }
    inline FreeBlockHeader* get_next_free_block(FreeBlockHeader* block) const noexcept {
        unsigned char* next_start = reinterpret_cast<unsigned char*>(block) + block->get_block_size();
        if(!within_boundary(next_start))
            return nullptr;
        FreeBlockHeader* next = reinterpret_cast<FreeBlockHeader*>(next_start);
        if(next->is_allocated())
            return nullptr;
        return next;
    }

    inline size_t get_alignment_offset(unsigned char* ptr, size_t alignment) const noexcept {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        return (alignment - (addr % alignment)) % alignment;
    }

    void merge(FreeBlockHeader* left, FreeBlockHeader* right) noexcept {
        remove_from_free_list(right);
        size_t new_size = left->get_block_size() + right->get_block_size();
        BlockFooter* footer = get_footer(reinterpret_cast<unsigned char*>(right), right->get_block_size());
        left->resize(new_size);
        footer->resize(new_size);
    }

    void perform_coalescence(FreeBlockHeader* block) noexcept {
        bool linked = false;
        FreeBlockHeader* before_block = get_prev_free_block(block);
        if(before_block) {
            merge(before_block, block);
            block = before_block;
            linked = true;
        }
        FreeBlockHeader* after_block = get_next_free_block(block);
        if(after_block)
            merge(block, after_block);
        if(!linked)
            push_front(block);
    }

    void split_block(FreeBlockHeader* block, size_t left_block_size, size_t right_block_size) noexcept {
        unsigned char* start = reinterpret_cast<unsigned char*>(block);
        size_t original_size = block->get_block_size();
        FreeBlockHeader* prev = block->ptrs.prev;
        FreeBlockHeader* next = block->ptrs.next;

        block->resize(left_block_size);
        new(get_footer(start, left_block_size)) BlockFooter(left_block_size, false);

        FreeBlockHeader* right = new(start + left_block_size) FreeBlockHeader(right_block_size, false);
        new(get_footer(start, original_size)) BlockFooter(right_block_size, false);

        // the right half takes the left half's place in the free list
        right->ptrs.prev = prev;
        right->ptrs.next = next;
        if(prev)
            prev->ptrs.next = right;
        if(next)
            next->ptrs.prev = right;
        if(block == head_)
            head_ = right;
        block->ptrs.next = nullptr;
        block->ptrs.prev = nullptr;
    }
};

template <typename T>
class CustomAllocator {
public:
    // stl typedefs
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    template <typename U>
    struct rebind {
        typedef CustomAllocator<U> other;
    };

    explicit CustomAllocator(size_t arena_size = INITIAL_ALLOCATOR_SIZE)
        : arena_(std::make_shared<Arena>(arena_size)) {}

    // copies and rebinds hand out memory from the same arena
    CustomAllocator(const CustomAllocator&) noexcept = default;
    CustomAllocator& operator=(const CustomAllocator&) noexcept = default;
    CustomAllocator(CustomAllocator&&) noexcept = default;
    CustomAllocator& operator=(CustomAllocator&&) noexcept = default;

    template <typename U>
    CustomAllocator(const CustomAllocator<U>& rhs) noexcept : arena_(rhs.arena_) {}

    T* allocate(size_t n) {
        if(n > SIZE_MAX / sizeof(T))
            throw std::bad_alloc();
        return static_cast<T*>(arena_->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T* p, size_t) noexcept {
        arena_->deallocate(p);
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }
    template <typename U>
    void destroy(U* p) {
        p->~U();
    }

    template <typename U>
    bool operator==(const CustomAllocator<U>& rhs) const noexcept {
        return arena_ == rhs.arena_;
    }

private:
    template <typename U> friend class CustomAllocator;

    std::shared_ptr<Arena> arena_;
};

struct alignas(32) Vec4 {
    double v[4];
};

static void test_allocate_and_deallocate() {
    CustomAllocator<int> alloc;
    int* p = alloc.allocate(4);
    assert(p);
    for(int i = 0; i < 4; ++i)
        alloc.construct(p + i, i * 3);
    for(int i = 0; i < 4; ++i)
        assert(p[i] == i * 3);
    for(int i = 0; i < 4; ++i)
        alloc.destroy(p + i);
    alloc.deallocate(p, 4);
}

static void test_reuse_after_free() {
    CustomAllocator<int> alloc;
    int* first = alloc.allocate(16);
    alloc.deallocate(first, 16);
    int* second = alloc.allocate(16);
    assert(first == second);
    alloc.deallocate(second, 16);
}

static void test_split_and_coalesce() {
    CustomAllocator<int> alloc;
    int* a = alloc.allocate(8);
    int* b = alloc.allocate(64);
    int* c = alloc.allocate(8);
    assert(a && b && c);

    // freeing b leaves a hole in the middle, the small request splits it and the
    // 1kb request has to come out of what is left at the tail
    alloc.deallocate(b, 64);
    int* small = alloc.allocate(2);
    int* big = alloc.allocate(256);
    assert(small && big);

    alloc.deallocate(small, 2);
    alloc.deallocate(big, 256);
    alloc.deallocate(a, 8);
    alloc.deallocate(c, 8);

    // nothing is live so coalescing should have put the arena back together
    int* whole = alloc.allocate(1000);
    assert(whole);
    alloc.deallocate(whole, 1000);
}

static void test_alignment() {
    CustomAllocator<Vec4> wide;
    Vec4* a = wide.allocate(1);
    Vec4* b = wide.allocate(2);
    assert(reinterpret_cast<uintptr_t>(a) % alignof(Vec4) == 0);
    assert(reinterpret_cast<uintptr_t>(b) % alignof(Vec4) == 0);
    wide.deallocate(a, 1);
    wide.deallocate(b, 2);

    // odd sizes used to leave block sizes that were neither aligned nor even, which
    // collided with the allocated flag living in the low bit
    CustomAllocator<char> chars;
    char* p1 = chars.allocate(1);
    char* p2 = chars.allocate(3);
    char* p3 = chars.allocate(7);
    assert(p1 != p2 && p2 != p3);
    *p1 = 'x';
    *p2 = 'y';
    *p3 = 'z';
    chars.deallocate(p2, 3);
    chars.deallocate(p1, 1);
    chars.deallocate(p3, 7);
}

static void test_exhaustion() {
    CustomAllocator<int> alloc(1024);
    std::vector<int*> blocks;
    bool ran_out = false;
    try {
        for(int i = 0; i < 1000; ++i)
            blocks.push_back(alloc.allocate(16));
    } catch(const std::bad_alloc&) {
        ran_out = true;
    }
    assert(ran_out);
    assert(!blocks.empty());

    for(int* p : blocks)
        alloc.deallocate(p, 16);
    int* again = alloc.allocate(16);
    assert(again);
    alloc.deallocate(again, 16);
}

static void test_stl_containers() {
    CustomAllocator<int> alloc(1 << 16);

    std::vector<int, CustomAllocator<int>> v(alloc);
    for(int i = 0; i < 200; ++i)
        v.push_back(i);
    assert(v.size() == 200);
    for(int i = 0; i < 200; ++i)
        assert(v[i] == i);

    std::vector<int, CustomAllocator<int>> copy = v;
    assert(copy == v);
    assert(copy.get_allocator() == v.get_allocator());
    v.clear();
    v.shrink_to_fit();
    assert(copy.size() == 200);

    // list allocates nodes, so this only works if rebinding does
    std::list<int, CustomAllocator<int>> l(alloc);
    for(int i = 0; i < 100; ++i)
        l.push_front(i);
    assert(l.size() == 100);
    assert(l.front() == 99 && l.back() == 0);
    l.clear();
}

// random alloc and free with a byte pattern in every block, so anything that hands
// out overlapping memory or loses a block shows up as a mismatch or an exhausted arena
static void test_random_stress() {
    constexpr size_t arena_size = 1 << 16;
    CustomAllocator<unsigned char> alloc(arena_size);
    std::mt19937 rng(20240607);
    std::vector<std::pair<unsigned char*, size_t>> live;

    for(int step = 0; step < 10000; ++step) {
        if(live.empty() || rng() % 100 < 60) {
            size_t n = 1 + rng() % 512;
            unsigned char* p = nullptr;
            try {
                p = alloc.allocate(n);
            } catch(const std::bad_alloc&) {
                continue;
            }
            std::memset(p, static_cast<int>(n & 0xff), n);
            live.emplace_back(p, n);
        }
        else {
            size_t i = rng() % live.size();
            unsigned char* p = live[i].first;
            size_t n = live[i].second;
            for(size_t j = 0; j < n; ++j)
                assert(p[j] == static_cast<unsigned char>(n & 0xff));
            alloc.deallocate(p, n);
            live[i] = live.back();
            live.pop_back();
        }
    }

    for(size_t i = 0; i < live.size(); ++i)
        alloc.deallocate(live[i].first, live[i].second);

    unsigned char* whole = alloc.allocate(arena_size - METADATA_SIZE_ALLOC);
    assert(whole);
    alloc.deallocate(whole, arena_size - METADATA_SIZE_ALLOC);
}

int main() {
    test_allocate_and_deallocate();
    test_reuse_after_free();
    test_split_and_coalesce();
    test_alignment();
    test_exhaustion();
    test_stl_containers();
    test_random_stress();

    std::cout << "all tests passed" << std::endl;
    return 0;
}
