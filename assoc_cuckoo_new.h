/* associative array */
void assoc3_init(const int hashpower_init);
item *assoc3_find(const char *key, const size_t nkey, const uint32_t hv);
int assoc3_insert(item *item, const uint32_t hv);
void assoc3_delete(const char *key, const size_t nkey, const uint32_t hv);
/* void do_assoc_move_next_bucket(void); */
/* int start_assoc_maintenance_thread(void); */
/* void stop_assoc_maintenance_thread(void); */

void assoc3_destroy(void);
void assoc3_pre_bench(void);
void assoc3_post_bench(void);
