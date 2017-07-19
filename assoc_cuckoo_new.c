/* -*- Mode: C; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*
 * Hash table implementation -- multi-reader/single-writer cuckoo hashing
 *
 */

//중복 되지 않았을까?
#include "memcached.h"

//
#include <sys/stat.h>

//
#include <sys/socket.h>

//
#include <sys/signal.h>

//
#include <sys/resource.h>

//
#include <fcntl.h>

//
#include <netinet/in.h>

//
#include <errno.h>

//
#include <stdlib.h>

//
#include <stdio.h>

//
#include <string.h>

//
#include <assert.h>

//
#include <pthread.h>

//
#include <xmmintrin.h>

//
#include <immintrin.h>

//
#include "memc3_config.h"

//
#include "memc3_util.h"

//
#include "bit_util.h"

//bucket lock
#if defined(MEMC3_LOCK_FINEGRAIN)

//이게 key version counter 개수 
#define fg_lock_count ((unsigned long int)1 << (13))

//mod 연산할 때 마스크
#define fg_lock_mask (fg_lock_count - 1)

//
static pthread_spinlock_t fg_locks[fg_lock_count];

//
static void fg_lock(uint32_t i1, uint32_t i2) {
    uint32_t j1, j2;
    j1 = i1 & fg_lock_mask;
    j2 = i2 & fg_lock_mask;
    if (j1 < j2) {
        pthread_spin_lock(&fg_locks[j1]);
        pthread_spin_lock(&fg_locks[j2]);
    } else if (j1 > j2) {
        pthread_spin_lock(&fg_locks[j2]);
        pthread_spin_lock(&fg_locks[j1]);
    } else
        pthread_spin_lock(&fg_locks[j1]);
}

//
static void fg_unlock(uint32_t i1, uint32_t i2) {
    uint32_t j1, j2;
    j1 = i1 & fg_lock_mask;
    j2 = i2 & fg_lock_mask;
    if (j1 < j2) {
        pthread_spin_unlock(&fg_locks[j2]);
        pthread_spin_unlock(&fg_locks[j1]);
    } else if (j1 > j2) {
        pthread_spin_unlock(&fg_locks[j1]);
        pthread_spin_unlock(&fg_locks[j2]);
    } else
        pthread_spin_unlock(&fg_locks[j1]);
}
#endif

/* 
 * Number of items in the hash table. 
 */
static unsigned int hash_items = 0;

/* 
 * Number of cuckoo kickouts. 
 */
static unsigned int num_moves  = 0;


/*
 * The maximum number of cuckoo operations per insert, 
 * we use 128 in the submission 
 * now change to 500

더 줄일 여지가 있음
줄일수록 좋음
 */
#define MAX_CUCKOO_COUNT 128


/*
 * the structure of a bucket
 변경할 여지가 있지만 아마 4개
 */
#define bucket_size 4

//
struct Bucket {
    //unsigned char = ub1 = TagType
    TagType   tags[bucket_size];      
    
	//Removed by Min
	//char      notused[4];

	//Added by Min
	//4바이트를 할당해놓았길래
	//2바이트 쓰고 2바이트 남겨놓음
	char      wall;			//어디까지가 h1인지 체크
	char      checkbit;		//슬롯별로 h몇인지 체크
	char      notused[2];	


    ValueType vals[bucket_size];
}  __attribute__((__packed__));

//전체 hash table
static struct Bucket* buckets;

//비었는지 확인하는 함수
#define IS_SLOT_EMPTY(i,j) (buckets[i].tags[j] == 0)

//#define IS_TAG_EQUAL(i,j,tag) ((buckets[i].tags[j] & tagmask) == tag)
//tag 일치하는지 확인하는 함수 
#define IS_TAG_EQUAL(bucket,j,tag) ((bucket.tags[j] & tagmask) == tag)

/*
 * Initialize the hash table
 */
//memcached.c 에서 사용 
void assoc3_init(const int hashtable_init) {
    //HASHPOWER_DEFAULT = 25
    hashpower = HASHPOWER_DEFAULT;
    if (hashtable_init) {
        hashpower = hashtable_init;
    }

    //ub4 = unsigned long int
    hashsize = (ub4) 1 << (hashpower);
    hashmask = hashsize - 1;

    /*
     * tagpower: number of bits per tag
     */
    tagpower = sizeof(TagType)*8;
    tagmask  = ((ub4) 1 << tagpower) - 1;

    //할당
    buckets = alloc(hashsize * sizeof(struct Bucket));
    //buckets = malloc(hashsize * sizeof(struct Bucket));
    if (! buckets) {
        fprintf(stderr, "Failed to init hashtable.\n");
        exit(EXIT_FAILURE);
    }
    memset(buckets, 0, sizeof(struct Bucket) * hashsize);

//
#ifdef MEMC3_LOCK_OPT
    memset(keyver_array, 0, sizeof(keyver_array));
#endif

//bucket lock인 듯하다
#ifdef MEMC3_LOCK_FINEGRAIN
    for (size_t i = 0; i < fg_lock_count; i++) {
        pthread_spin_init(&fg_locks[i], PTHREAD_PROCESS_PRIVATE);
    }
#endif

    //
    STATS_LOCK();
    stats.hash_power_level = hashpower;
    stats.hash_bytes = hashsize * sizeof(struct Bucket);
    STATS_UNLOCK();
}

/*
 * Destroy all the buckets
 */
//안 쓰는데?
void assoc3_destroy() {
    dealloc(buckets);
}


/*
 * Try to read bucket i and check if the given tag is there
 */
//??????????????????????????????????????????
//assoc3_find()에서 사용인데 주석 처리 되어있음 
//결국은 안 쓴다는 얘기인데 
static __attribute__ ((unused))
item *try_read(const char *key, const size_t nkey, TagType tag, size_t i) {

#ifdef MEMC3_ENABLE_TAG
    volatile uint32_t tmp = *((uint32_t *) &(buckets[i]));
#endif

    //bucket_size = 4
    for (size_t j = 0; j < bucket_size; ++j) {
#ifdef MEMC3_ENABLE_TAG
        //if (IS_TAG_EQUAL(buckets[i], j, tag))
        //tag가 일치하면
        uint8_t ch = ((uint8_t*) &tmp)[j];
        if (ch == tag)
#endif
        {
            /* volatile __m128i p, q; */
            /* p = _mm_loadu_si128((__m128i const *) &buckets[i].vals[0]); */
            /* q = _mm_loadu_si128((__m128i const *) &buckets[i].vals[2]); */
            /* item *vals[4]; */

            /* _mm_storeu_si128((__m128i *) vals, p); */
            /* _mm_storeu_si128((__m128i *) (vals + 2), q); */
            /* item *it = vals[j]; */

            item *it = buckets[i].vals[j];
            
#ifndef MEMC3_ENABLE_TAG
            if (it == NULL)
                return NULL;
#endif
            //key 비교하고 일치하면 return
            char* it_key = (char*) ITEM_key(it);
            if (keycmp(key, it_key, nkey)) {
                return it;
            }
        }
    }
    return NULL;
}


/*
 * The interface to find a key in this hash table
 */
//items.c  do_item_get()에서 사용 
item *assoc3_find(const char *key, const size_t nkey, const uint32_t hv) {
    //해시값으로 tag 계산
    //첫번째 버킷, 두번째 버킷 계산
    TagType tag = _tag_hash(hv);
    size_t i1   = _index_hash(hv);
    size_t i2   = _alt_index(i1, tag);

    item *result = NULL;

//optimistic lock
#ifdef MEMC3_LOCK_OPT
    //i1, i2 중에 작은 걸 lock
    //왜 더 작은 걸?
    size_t lock = _lock_index(i1, i2, tag);
    uint32_t vs, ve;
TryRead:
    //key counter version을 읽어오라고
    //atomic 연산으로 0 더해서 가져와 
    //vs = version start ? 
    vs = read_keyver(lock);
#endif

//bucket lock인 듯하다
#ifdef MEMC3_LOCK_FINEGRAIN
    fg_lock(i1, i2);
#endif

    //

    //_mm_prefetch(&buckets[i2], _MM_HINT_NTA);

    /* item *r1, *r2; */
    /* r1 = try_read(key, nkey, tag, i1); */
    /* r2 = try_read(key, nkey, tag, i2); */
    /* if (r1) result = r1; */
    /* else result = r2; */


#ifdef MEMC3_ENABLE_TAG
    //Removed by Min
    //volatile uint32_t tags1, tags2;
    //tags1 = *((uint32_t *) &(buckets[i1]));
    //tags2 = *((uint32_t *) &(buckets[i2]));

    //Added by Min
    volatile uint64_t tags1, tags2;
    tags1 = *((uint64_t *) &(buckets[i1]));
    tags2 = *((uint64_t *) &(buckets[i2]));
#endif


    //첫번째 버킷에서 찾기
    //Removed by Min
/*    for (size_t j = 0; j < 4; j ++) {
#ifdef MEMC3_ENABLE_TAG
        uint8_t ch = ((uint8_t*) &tags1)[j];
        if (ch == tag)
#endif
        {
            item *it = buckets[i1].vals[j];
            
//#ifndef MEMC3_ENABLE_TAG
            if (it == NULL)
                continue;
//#endif
            char* it_key = (char*) ITEM_key(it);
            if (keycmp(key, it_key, nkey)) {
                result = it;
                break;
            }
        }
    }
*/

    //Added by Min
    //wall까지만 find
    //h1로 들어온 데까지만 find
    uint8_t wall = ((uint8_t*) &tags1)[4];
    for (size_t j = 0; j < wall; ++j) {
#ifdef MEMC3_ENABLE_TAG
        uint8_t ch = ((uint8_t*) &tags1)[j];
        if (ch == tag)
#endif
        {
            item *it = buckets[i1].vals[j];
            
            if (it == NULL)
                continue;

            char* it_key = (char*) ITEM_key(it);
            if (keycmp(key, it_key, nkey)) {
                result = it;
                break;
            }
        }
    }




    //두번째 버킷에서 찾기
    if (!result) 
    {
        //Removed by Min
/*       for (size_t j = 0; j < 4; j ++) {
#ifdef MEMC3_ENABLE_TAG

            uint8_t ch = ((uint8_t*) &tags2)[j];
            if (ch == tag)
#endif
            {

                item *it = buckets[i2].vals[j];
            
//#ifndef MEMC3_ENABLE_TAG
                if (it == NULL)
                    continue;
//#endif
                char* it_key = (char*) ITEM_key(it);
                if (keycmp(key, it_key, nkey)) {
                    result = it;
                    break;
                }
            }
        }
        */

        //Added by Min
        wall = ((uint8_t*) &tags2)[4];
        for (size_t j = wall; j < bucket_size; j ++) {
#ifdef MEMC3_ENABLE_TAG
            uint8_t ch = ((uint8_t*) &tags2)[j];
            if (ch == tag)
#endif
            {
                item *it = buckets[i2].vals[j];
            
                if (it == NULL)
                    continue;

                char* it_key = (char*) ITEM_key(it);
                if (keycmp(key, it_key, nkey)) {
                    result = it;
                    break;
                }
            }
        }
    }

    //result = try_read(key, nkey, tag, i1);
    //if (!result)
    //{
    //   result = try_read(key, nkey, tag, i2);
    //}


//opt lock
#ifdef MEMC3_LOCK_OPT
    //ve = version end
    ve = read_keyver(lock);

    //홀수거나? version count가 다르면 반복 
    if (vs & 1 || vs != ve)
        goto TryRead;
#endif

//btk lock
#ifdef MEMC3_LOCK_FINEGRAIN
    fg_unlock(i1, i2);
#endif

    return result;
}


/* 
 * Make bucket  from[idx] slot[whichslot] available to insert a new item
 * return idx on success, -1 otherwise
 * @param from:   the array of bucket index
 * @param whichslot: the slot available
 * @param  depth: the current cuckoo depth
 */

//MAX_CUCKOO_COUNT = 500
//MEMC3_ASSOC_CUCKOO_WIDTH = 1 아마 path 개수인듯

//중복되지 않을까?
size_t    cp_buckets[MAX_CUCKOO_COUNT][MEMC3_ASSOC_CUCKOO_WIDTH];
size_t    cp_slots[MAX_CUCKOO_COUNT][MEMC3_ASSOC_CUCKOO_WIDTH];
ValueType cp_vals[MAX_CUCKOO_COUNT][MEMC3_ASSOC_CUCKOO_WIDTH];

//Added by Min
size_t    cp_bits[MAX_CUCKOO_COUNT][MEMC3_ASSOC_CUCKOO_WIDTH];

int       kick_count = 0;

//cp = Cuckoo Path
//cuckoo()에서 사용 
static int cp_search(size_t depth_start, size_t *cp_index) {

    //MAX_CUCKOO_COUNT = 128
    int depth = depth_start;
    while ((kick_count < MAX_CUCKOO_COUNT) && 
           (depth >= 0) && 
           (depth < MAX_CUCKOO_COUNT - 1)) {

        size_t *from  = &(cp_buckets[depth][0]);
        size_t *to    = &(cp_buckets[depth + 1][0]);
        
        /*
         * Check if any slot is already free
         */
        //Removed by Min
        //for (size_t idx = 0; idx < MEMC3_ASSOC_CUCKOO_WIDTH; idx ++) 
        {
            //Removed by Min
            //size_t i = from[idx];

            //Added by Min
            size_t i = from[0];

            size_t j;
            for (j = 0; j < bucket_size; ++j) {
                if (IS_SLOT_EMPTY(i, j)) {
                    //Removed by Min
                    //cp_slots[depth][idx] = j;

                    //Added by Min
                    cp_slots[depth][0] = j;

                    //cp_vals[depth][idx]  = buckets[i].vals[j];

                    //Removed by Min
                    //*cp_index   = idx;

                    //Added by Min
                    *cp_index   = 0;

                    return depth;
                }
            }

            //Removed by Min
            //j          = rand() % bucket_size; // pick the victim item
            
            //Added by Min
            volatile uint64_t tags;
            tags = *((uint64_t *) &(buckets[i]));
            uint8_t bits = ((uint8_t*) &tags)[5];
            uint8_t min  = (bits & 0x03);
            uint8_t tmp;
            j = 0;
            for (size_t k = 1; k < bucket_size; ++k) {
                tmp = (bits >> (2 * k)) & 0x03;
                if(min > tmp) {
                    min = tmp;
                    j = k;
                }
            }

            //Removed by Min
            //cp_slots[depth][idx] = j;
            //cp_vals[depth][idx]  = buckets[i].vals[j];
            //to[idx]    = _alt_index(i, buckets[i].tags[j]);

            //Added by Min 
            cp_slots[depth][0] = j;
            cp_vals[depth][0]  = buckets[i].vals[j];
            to[0]    = _alt_index(i, buckets[i].tags[j]);
            cp_bits[depth][0]  = min;

        }

        //Removed by Min
        //kick_count += MEMC3_ASSOC_CUCKOO_WIDTH;

        //Added by Min
        ++kick_count;

        ++depth;
    }

    printf("%u max cuckoo achieved, abort\n", kick_count);
    return -1;
}

//cuckoo() 에서 사용 
static int cp_backmove(size_t depth_start, size_t idx) {
    
    int depth = depth_start;
    while (depth > 0) {
        size_t i1 = cp_buckets[depth - 1][idx];
        size_t i2 = cp_buckets[depth][idx];
        size_t j1 = cp_slots[depth - 1][idx];
        size_t j2 = cp_slots[depth][idx];

        /*
         * We plan to kick out j1, but let's check if it is still there;
         * there's a small chance we've gotten scooped by a later cuckoo.
         * If that happened, just... try again.
         */
        //이거 생각해 봐야 할 듯 
        if (buckets[i1].vals[j1] != cp_vals[depth - 1][idx]) {
            /* try again */
            return depth;
        }

        //에러 체크 
        assert(IS_SLOT_EMPTY(i2,j2));

//opt lock
//atomic 하게 1 더함 
#ifdef MEMC3_LOCK_OPT
        size_t lock   = _lock_index(i1, i2, 0);
        incr_keyver(lock);
#endif

//bkt lock 
#ifdef MEMC3_LOCK_FINEGRAIN
        fg_lock(i1, i2);
#endif
    
        //Added by Min
        //wall 고려해봐야 함 
        volatile uint64_t tags1, tags2;
        tags1 = *((uint64_t *) &(buckets[i1]));
        tags2 = *((uint64_t *) &(buckets[i2]));
        uint8_t bits2 = ((uint8_t*) &tags2)[5];
        uint8_t b1 = cp_bits[depth - 1][idx];
        if(b1 != 0x03) {
            ++b1;
        }

        buckets[i2].tags[j2] = buckets[i1].tags[j1];
        buckets[i2].vals[j2] = buckets[i1].vals[j1];

        //Added by Min
        bits2 &= ((0xFC << (2 * j2)) | (0xFC >> (8 - (2 * j2))));
        bits2 |= (b1 << (2 * j2));

        buckets[i1].tags[j1] = 0;
        buckets[i1].vals[j1] = 0;

#ifdef PRINT_LF
        ++num_moves;
#endif

//opt unlock
#ifdef MEMC3_LOCK_OPT
        incr_keyver(lock);
#endif

//bkt unlock
#ifdef MEMC3_LOCK_FINEGRAIN
        fg_unlock(i1, i2);
#endif
        --depth;
    }
    return depth;
}

//assoc3_insert() 에서 사용 
static int cuckoo(int depth) {
    int cur;
    size_t idx;

    kick_count = 0;    
    while (1) {

        cur = cp_search(depth, &idx);
        if (cur < 0)
            return -1;
        assert(idx >= 0);
        cur = cp_backmove(cur, idx);
        if (cur == 0) {
            return idx;
        }

        depth = cur - 1;
    }

    return -1;
}

/*
 * Try to add an item to bucket i, 
 * return true on success and false on failure
 */
//assoc3_insert() 에서 사용 
static bool try_add(item* it, TagType tag, size_t i, size_t lock) {

//LF = Load Factor
#ifdef PRINT_LF
    static double next_lf = 0.0;
#endif

        //Added by Min
        //wall 고려해봐야 함 
        volatile uint64_t tags;
        tags = *((uint64_t *) &(buckets[i]));
        uint8_t bits = ((uint8_t*) &tags)[5];

    for (size_t j = 0; j < bucket_size; ++j) {
        if (IS_SLOT_EMPTY(i, j)) {

//opt lock
#ifdef MEMC3_LOCK_OPT
            incr_keyver(lock);
#endif

//bkt lock 
#ifdef MEMC3_LOCK_FINEGRAIN
            fg_lock(i, i);
#endif

            buckets[i].tags[j] = tag;
            buckets[i].vals[j] = it;
            
            //Added by Min
            bits &= ((0xFC << (2 * j)) | (0xFC >> (8 - (2 * j))));
            bits |= (0x01 << (2 * j2));

            /* atomic add for hash_item */
            //__sync_fetch_and_add(&hash_items, 1);
            ++hash_items;

#ifdef PRINT_LF
            if ((hash_items & 0x0fff) == 1) {
                double lf = (double) hash_items / bucket_size / hashsize;
                if (lf > next_lf) {
                    struct timeval tv; 
                    gettimeofday(&tv, NULL);
                    double tvd_now = (double)tv.tv_sec + (double)tv.tv_usec/1000000;
                    
                    printf("loadfactor=%.4f\tmoves=%u\thash_items=%u\ttime=%.8f\n", lf, num_moves, hash_items, tvd_now);
                    next_lf += 0.05;
                }
            }
#endif

//opt unlock
#ifdef MEMC3_LOCK_OPT
            incr_keyver(lock);
#endif

//bkt unlock
#ifdef MEMC3_LOCK_FINEGRAIN
            fg_unlock(i, i);
#endif

            return true;
        }
    }
    return false;
}



/* Note: this isn't an assoc_update.  The key must not already exist to call this */
// need to be protected by cache_lock

//items.c  do_item_link_nolock 에 쓰임 
int assoc3_insert(item *it, const uint32_t hv) {

    TagType tag = _tag_hash(hv);
    size_t i1   = _index_hash(hv);
    size_t i2   = _alt_index(i1, tag);
    size_t lock = _lock_index(i1, i2, tag);

    //첫번째 버킷에 넣어봐 
    if (try_add(it, tag, i1, lock))
        return 1;

    //두번째 버킷에 넣어봐 
    if (try_add(it, tag, i2, lock))
        return 1;

    //Removed by Min
    //int idx;

    //Added by Min
    int idx = 0;
    size_t depth = 0;

    //Removed by Min
    //for (idx = 0; idx < MEMC3_ASSOC_CUCKOO_WIDTH; idx ++) 
    {
        if (rand() % 2) 
            cp_buckets[depth][idx] = i1;
        else
            cp_buckets[depth][idx] = i2;
    }
    size_t j;
    idx = cuckoo(depth);
    if (idx >= 0) {
        i1 = cp_buckets[depth][idx];
        j = cp_slots[depth][idx];

        //이거 뭐야??????????????????
        if (buckets[i1].vals[j] != 0)
            printf("ououou\n");

        //넣는다 
        if (try_add(it, tag, i1, lock))
            return 1;

        //프린트 하지도 않으면서 
        printf("mmm i1=%zu i=%d\n", i1, idx);
    }

    //cuckoo path 못 찾으면 다 찼다고 
    printf("hash table is full (hashpower = %d, hash_items = %u, load factor = %.2f), need to increase hashpower\n",
           hashpower, hash_items, 1.0 * hash_items / bucket_size / hashsize);
    return 0;
}    


//assoc3_delete() 에서 사용 
static bool try_del(const char*key, const size_t nkey, TagType tag, size_t i, size_t lock) {
    for (size_t j = 0; j < bucket_size; j ++) {
#ifdef MEMC3_ENABLE_TAG
        //if (IS_TAG_EQUAL(i, j, tag))
        if (IS_TAG_EQUAL(buckets[i], j, tag))
#endif
        {
            item *it = buckets[i].vals[j];

#ifndef MEMC3_ENABLE_TAG
            if (it == NULL)
                return false;
#endif

            if (keycmp(key, ITEM_key(it), nkey)) {

#ifdef MEMC3_LOCK_OPT
                incr_keyver(lock);
#endif

#ifdef MEMC3_LOCK_FINEGRAIN
                fg_lock(i, i);
#endif

                buckets[i].tags[j] = 0;
                buckets[i].vals[j] = 0;
                hash_items --;

#ifdef MEMC3_LOCK_OPT
                incr_keyver(lock);
#endif

#ifdef MEMC3_LOCK_FINEGRAIN
                fg_unlock(i, i);
#endif

                return true;
            }
        }
    }
    return false;
}

// need to be protected by cache_lock
// items.c  do_item_unlink_nolock() 에서 사용 
void assoc3_delete(const char *key, const size_t nkey, const uint32_t hv) {

    TagType tag = _tag_hash(hv);
    size_t   i1 = _index_hash(hv);
    size_t   i2 = _alt_index(i1, tag);
    size_t lock = _lock_index(i1, i2, tag);


    if (try_del(key, nkey, tag, i1, lock))
        return;

    if (try_del(key, nkey, tag, i2, lock))
        return;


    /* Note:  we never actually get here.  the callers don't delete things
       they can't find. */
    assert(false);
}


//not used
void assoc3_pre_bench() {
    num_moves = 0;
}

//not used 
void assoc3_post_bench() {

    size_t total_size = 0;

    printf("hash_items = %u\n", hash_items);
    printf("index table size = %zu\n", hashsize);
    printf("hashtable size = %zu KB\n",hashsize*sizeof(struct Bucket)/1024);
    printf("hashtable load factor= %.5f\n", 1.0 * hash_items / bucket_size / hashsize);
    total_size += hashsize*sizeof(struct Bucket);
    printf("total_size = %zu KB\n", total_size / 1024);
#ifdef PRINT_LF
    printf("moves per insert = %.2f\n", (double) num_moves / hash_items);
#endif
    printf("\n");
}

