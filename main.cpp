#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <exception>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <new>
#include <mutex>

// memory allocator
// implement a template class customallocator<t> that's compatible with stl containers
// supports small (<256 bytes) and large allocations with proper alignment
// required functions:
// allocate(size_t n)
// deallocate(t* p, size_t n)
// construct(t* p, args&&... args)
// destroy(t* p)
// must be stateless, handle alignment, be exception-safe, propagate to template params, support rebinding
// x o x x x x x x x
// default allocate 4096 bytes
// handle coalesce
// explicit free list

struct BlockHeader; // forward decl

constexpr size_t ALIGNMENT = alignof(std::max_align_t);

constexpr size_t align_up(size_t n) {
    return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

struct BlockFooter {
    size_t block_size;

    // block size is multiple of 2 so lsb is free for allocated flag
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
    // out of line def below
    size_t get_payload_size() const;
};

struct BlockHeader {
    size_t block_size;
    size_t padding;

    // block size is multiple of 2 so lsb is free for allocated flag
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
    // out of line def below
    size_t get_payload_size() const;
    unsigned char* get_payload_ptr() const {
        return reinterpret_cast<unsigned char*>(const_cast<BlockHeader*>(this)) + sizeof(BlockHeader) + padding;
    }
};

size_t BlockHeader::get_payload_size() const {
    return get_block_size() - sizeof(BlockHeader) - sizeof(BlockFooter);
}

size_t BlockFooter::get_payload_size() const {
    return get_block_size() - (sizeof(BlockHeader) + sizeof(BlockFooter));
}

struct FreeBlockHeader {
    size_t block_size;
    struct {
        FreeBlockHeader* next;
        FreeBlockHeader* prev;
    } ptrs;

    // block size is multiple of 2 so lsb is free for allocated flag
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
    size_t get_payload_size() const {
        return get_block_size() - (sizeof(FreeBlockHeader) + sizeof(BlockFooter));
    }
    unsigned char* get_payload_ptr() const {
        return reinterpret_cast<unsigned char*>(const_cast<FreeBlockHeader*>(this)) + sizeof(FreeBlockHeader);
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

    template <typename U>
    struct rebind {
        typedef CustomAllocator<U> other;
    };

    CustomAllocator()
      : data_(reinterpret_cast<unsigned char*>(mmap(nullptr, INITIAL_ALLOCATOR_SIZE,
                                                     PROT_READ | PROT_WRITE,
                                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0))),
        size_(INITIAL_ALLOCATOR_SIZE)
    {
        if(data_ == MAP_FAILED)
            throw std::bad_alloc();
        // make one free block for whole region
        head_ = new(data_) FreeBlockHeader(size_, false);
        new(data_ + size_ - sizeof(BlockFooter)) BlockFooter(size_, false);
    }
    ~CustomAllocator() {
        if(data_)
            munmap(data_, size_);
        data_ = nullptr;
        head_ = nullptr;
        size_ = 0;
    }
    CustomAllocator(const CustomAllocator& rhs) = delete;
    CustomAllocator& operator=(const CustomAllocator& rhs) = delete;

    CustomAllocator(CustomAllocator&& rhs) noexcept
      : data_(std::exchange(rhs.data_, nullptr)),
        head_(std::exchange(rhs.head_, nullptr)),
        size_(std::exchange(rhs.size_, 0))
    {}
    CustomAllocator& operator=(CustomAllocator&& rhs) noexcept {
        if(this != &rhs) {
            if(data_)
                munmap(data_, size_);
            data_ = std::exchange(rhs.data_, nullptr);
            head_ = std::exchange(rhs.head_, nullptr);
            size_ = std::exchange(rhs.size_, 0);
        }
        return *this;
    }

    // allocate n objects of type t
    T* allocate(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(n > SIZE_MAX / sizeof(T))
            throw std::bad_alloc();
        size_t required_size = n * sizeof(T);
        size_t alignment = alignof(T) < ALIGNMENT ? ALIGNMENT : alignof(T);
        if(!head_)
            throw std::bad_alloc();
        if(required_size > size_ - METADATA_SIZE_ALLOC)
            throw std::bad_alloc();

        // block sizes are rounded so that every header and footer lands on an aligned
        // address and the low bit of the size stays free for the allocated flag
        size_t padding_needed = 0;
        size_t allocated_block_size = 0;
        FreeBlockHeader* curr_block = head_;
        while(curr_block) {
            unsigned char* start = reinterpret_cast<unsigned char*>(curr_block);
            size_t padding = get_alignment_offset(start + sizeof(BlockHeader), alignment);
            size_t total = align_up(METADATA_SIZE_ALLOC + padding + required_size);
            if(total <= curr_block->get_block_size()) {
                padding_needed = padding;
                allocated_block_size = total;
                break;
            }
            curr_block = curr_block->ptrs.next;
        }
        if(!curr_block)
            throw std::bad_alloc();

        size_t original_size = curr_block->get_block_size();

        // split free block if enough room
        if(original_size - allocated_block_size >= MINIMUM_BLOCK_SIZE)
            split_block(curr_block, allocated_block_size, original_size - allocated_block_size);
        else {
            remove_from_free_list(curr_block);
            allocated_block_size = original_size;
        }

        // mark block as allocated and set padding
        BlockHeader* block = new(reinterpret_cast<void*>(curr_block)) BlockHeader(allocated_block_size, true);
        block->padding = padding_needed;
        BlockFooter* footer = get_block_footer(block);
        new(footer) BlockFooter(allocated_block_size, true);

        // an over aligned t pushes the payload past the header, so leave the distance
        // back to the header in the word right in front of the payload. with no padding
        // that word is the header's own padding field, otherwise it lands in the gap
        unsigned char* payload = block->get_payload_ptr();
        *reinterpret_cast<size_t*>(payload - sizeof(size_t)) = padding_needed;
        return reinterpret_cast<T*>(payload);
    }

    void deallocate(T* p) {
        std::lock_guard<std::mutex> lock(mutex_);
        unsigned char* ptr = reinterpret_cast<unsigned char*>(p);
        if(ptr < data_ || ptr >= (data_ + size_))
            throw std::logic_error("pointer to deallocate is outside bounds");
        
        BlockHeader* block = get_block_header(p);
        size_t block_size = block->get_block_size();
        get_block_footer(block)->change_allocation(false);
        // convert allocated block to free block
        FreeBlockHeader* free_block = new(reinterpret_cast<void*>(block))
                                     FreeBlockHeader(block_size, false);

        perform_coalescence(free_block);
    }

    // stl construct/destroy
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }
    template <typename U>
    void destroy(U* p) {
        p->~U();
    }

private:
    std::mutex mutex_;
    unsigned char* data_;
    FreeBlockHeader* head_;
    size_t size_;

    void remove_from_free_list(FreeBlockHeader* block) {
        if(block == head_)
            head_ = block->ptrs.next;
        if(block->ptrs.prev)
            block->ptrs.prev->ptrs.next = block->ptrs.next;
        if(block->ptrs.next)
            block->ptrs.next->ptrs.prev = block->ptrs.prev;
        block->ptrs.next = nullptr;
        block->ptrs.prev = nullptr;
    }

    inline bool within_boundary(unsigned char* ptr) const noexcept {
        return (ptr >= data_) && (ptr < data_ + size_);
    }

    inline FreeBlockHeader* get_prev_block(FreeBlockHeader* header) const noexcept {
        unsigned char* footer_location = reinterpret_cast<unsigned char*>(header) - sizeof(BlockFooter);
        if(!within_boundary(footer_location))
            return nullptr;
        BlockFooter* prev_footer = reinterpret_cast<BlockFooter*>(footer_location);
        size_t prev_block_size = prev_footer->get_block_size();
        unsigned char* prev_header_ptr = reinterpret_cast<unsigned char*>(header) - prev_block_size;
        if(!within_boundary(prev_header_ptr))
            return nullptr;
        return reinterpret_cast<FreeBlockHeader*>(prev_header_ptr);
    }
    inline FreeBlockHeader* get_next_block(FreeBlockHeader* header) const noexcept {
        unsigned char* header_location = reinterpret_cast<unsigned char*>(header) + header->get_block_size();
        if(!within_boundary(header_location))
            return nullptr;
        return reinterpret_cast<FreeBlockHeader*>(header_location);
    }
    inline BlockFooter* get_block_footer(BlockHeader* header) const noexcept {
        return reinterpret_cast<BlockFooter*>(
            reinterpret_cast<unsigned char*>(header) + (header->get_block_size() - sizeof(BlockFooter))
        );
    }
    inline BlockFooter* get_block_footer(FreeBlockHeader* header) const noexcept {
        return reinterpret_cast<BlockFooter*>(
            reinterpret_cast<unsigned char*>(header) + (header->get_block_size() - sizeof(BlockFooter))
        );
    }
    inline BlockHeader* get_block_header(T* ptr) const noexcept {
        unsigned char* payload = reinterpret_cast<unsigned char*>(ptr);
        size_t padding = *reinterpret_cast<size_t*>(payload - sizeof(size_t));
        return reinterpret_cast<BlockHeader*>(payload - sizeof(BlockHeader) - padding);
    }

    inline size_t get_alignment_offset(unsigned char* ptr, size_t alignment) const noexcept {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        size_t offset = (alignment - (addr % alignment)) % alignment;
        return offset;
    }

    void merge(FreeBlockHeader* left, FreeBlockHeader* right) noexcept {
        remove_from_free_list(right);
        size_t new_size = left->get_block_size() + right->get_block_size();
        BlockFooter* footer = get_block_footer(right);
        left->resize(new_size);
        footer->resize(new_size);
    }

    void push_front(FreeBlockHeader* block) noexcept {
        block->ptrs.prev = nullptr;
        block->ptrs.next = head_;
        if(head_)
            head_->ptrs.prev = block;
        head_ = block;
    }

    void perform_coalescence(FreeBlockHeader* block) noexcept {
        // merging backwards keeps the block that was already on the free list, only
        // a block that did not merge that way still needs linking in
        bool linked = false;
        FreeBlockHeader* before_block = get_prev_block(block);
        if(before_block && !before_block->is_allocated()) {
            merge(before_block, block);
            block = before_block;
            linked = true;
        }
        FreeBlockHeader* after_block = get_next_block(block);
        if(after_block && !after_block->is_allocated())
            merge(block, after_block);
        if(!linked)
            push_front(block);
    }

    void split_block(FreeBlockHeader* block, size_t left_block_size, size_t right_block_size) {
        unsigned char* start = reinterpret_cast<unsigned char*>(block);
        size_t original_size = block->get_block_size();
        FreeBlockHeader* prev = block->ptrs.prev;
        FreeBlockHeader* next = block->ptrs.next;

        BlockFooter* left_footer = reinterpret_cast<BlockFooter*>(start + left_block_size - sizeof(BlockFooter));
        new(left_footer) BlockFooter(left_block_size, false);
        // update the left block header with new size
        block->resize(left_block_size);

        FreeBlockHeader* new_block_header = new(start + left_block_size) FreeBlockHeader(right_block_size, false);
        BlockFooter* right_footer = reinterpret_cast<BlockFooter*>(start + original_size - sizeof(BlockFooter));
        new(right_footer) BlockFooter(right_block_size, false);

        // the right half takes the left half's place in the list, it needs the old
        // links as well or everything past it drops off the free list
        new_block_header->ptrs.prev = prev;
        new_block_header->ptrs.next = next;
        if(prev)
            prev->ptrs.next = new_block_header;
        if(next)
            next->ptrs.prev = new_block_header;
        if(block == head_)
            head_ = new_block_header;
        block->ptrs.next = nullptr;
        block->ptrs.prev = nullptr;
    }
};

int main() {
    CustomAllocator<int> alloc;
    try {
        int* int_ptr = alloc.allocate(220);
        new(int_ptr) int(2);
        std::cout << *int_ptr << std::endl;
    } catch(...) {
        std::exception_ptr e = std::current_exception();
    }

    int* int2_s = alloc.allocate(32);
    new(int2_s) int(5);
    std::cout << *int2_s << std::endl;

    return 0;
}
