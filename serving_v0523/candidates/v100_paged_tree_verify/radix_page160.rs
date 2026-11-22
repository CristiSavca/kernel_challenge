// Include from the pinned crate's tests/unified_tree_core.rs to exercise its real core.
const RADIX_TEST_PAGE: usize = 160;

fn radix_page160_slots(pages: &[i64]) -> Vec<i64> {
    let size = RADIX_TEST_PAGE as i64;
    pages.iter().flat_map(|p| p * size..(p + 1) * size).collect()
}

fn radix_page160_expect(
    core: &mut UnifiedTreeCore<Vec<i64>>,
    key: &[i64],
    slots: &[i64],
) {
    let result = core.match_prefix(&match_params(&key.to_vec()));
    assert_eq!(result.device_prefix_len, slots.len());
    assert!(matched_device_indices(core, &result).equal(&Tensor::from_slice(slots)));
}

#[test]
fn radix_page160_branch_and_physical_page_ownership() {
    let original: Vec<i64> = (0..640).collect();
    let original_slots = radix_page160_slots(&[27, 3, 80, 6]);
    let new_slots = radix_page160_slots(&[5, 61, 9, 31]);
    for &fork in &[0usize, 1, 158, 159, 160, 161, 318, 319, 320, 321, 478, 479, 480, 481, 638, 639] {
        let mut core = events_core(RADIX_TEST_PAGE);
        core.insert(&insert_params(&original, &original_slots));
        let common = fork / RADIX_TEST_PAGE * RADIX_TEST_PAGE;
        let mut branch = original.clone();
        branch[fork] += 10000;
        radix_page160_expect(&mut core, &branch, &original_slots[..common]);
        let inserted = core.insert(&insert_params(&branch, &new_slots));
        assert_eq!(inserted.prefix_len, common);
        let mut expected = original_slots[..common].to_vec();
        expected.extend_from_slice(&new_slots[common..]);
        radix_page160_expect(&mut core, &branch, &expected);
        radix_page160_expect(&mut core, &original, &original_slots);
        for len in [0usize, 1, 159, 160, 161, 319, 320, 321, 479, 480, 639, 640] {
            let aligned = len / RADIX_TEST_PAGE * RADIX_TEST_PAGE;
            radix_page160_expect(&mut core, &branch[..len], &expected[..aligned]);
        }
        core.sanity_check(&[], &[]);
    }
}

#[test]
fn radix_page160_full_host_split_reload_and_physical_reuse() {
    let key: Vec<i64> = (1000..1640).collect();
    let original = radix_page160_slots(&[7, 19, 3, 42]);
    let host = radix_page160_slots(&[10, 8, 79, 1]);
    let replacement = radix_page160_slots(&[41, 2, 51, 33]);
    let mut core = events_core(RADIX_TEST_PAGE);
    core.set_hicache_enabled();
    let leaf = core.insert(&insert_params(&key, &original)).last_device_node_id.unwrap();
    let (backup, components) = core.build_backup_spec(leaf).unwrap();
    assert!(backup.equal(&Tensor::from_slice(&original)));
    assert!(components.is_empty());
    core.commit_backup(leaf, Tensor::from_slice(&host), HashMap::new()).unwrap();
    let (_, freed) = core.evict_device_leaf(leaf, false).unwrap();
    assert_eq!(freed.tracker[&FULL], key.len());
    assert!(freed.device_frees[&FULL][0].equal(&Tensor::from_slice(&original)));
    let unrelated: Vec<i64> = (5000..5640).collect();
    core.insert(&insert_params(&unrelated, &original));
    radix_page160_expect(&mut core, &unrelated, &original);
    let mut fork = key.clone();
    fork[321] += 13;
    let partial = core.match_prefix(&match_params(&fork));
    assert_eq!(partial.device_prefix_len, 0);
    assert_eq!(partial.host_hit_length, 320);
    let split_node = partial.last_host_node_id;
    let host_match = core.match_prefix(&match_params(&key));
    assert_eq!(host_match.device_prefix_len, 0);
    assert_eq!(host_match.host_hit_length, key.len());
    let (xfer, components) = core.build_load_back_spec(leaf, None).unwrap();
    assert!(xfer.host_indices.as_ref().unwrap().equal(&Tensor::from_slice(&host)));
    assert_eq!(xfer.nodes_to_load.as_ref().unwrap(), &[split_node, leaf]);
    core.commit_load_back(leaf, Tensor::from_slice(&replacement), xfer, components).unwrap();
    let lock = core.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
    core.finish_load_back(leaf).unwrap();
    radix_page160_expect(&mut core, &key, &replacement);
    radix_page160_expect(&mut core, &fork, &replacement[..320]);
    radix_page160_expect(&mut core, &unrelated, &original);
    core.dec_lock_ref(leaf, &lock.to_dec_params(), false).unwrap();
    core.sanity_check(&[], &[]);
}

#[test]
fn radix_page160_event_block_chain_continues_at_shared_boundary() {
    let base: Vec<i64> = (0..640).collect();
    let mut branch = base.clone();
    branch[319] = 999;
    let mut core = events_core(RADIX_TEST_PAGE);
    for (key, pages, shared_pages) in [(&base, [19, 7, 22, 4], 0usize), (&branch, [11, 35, 23, 6], 1)] {
        let slots = radix_page160_slots(&pages);
        core.insert(&insert_params(key, &slots));
        let hashes: Vec<_> = crate::node::get_hash_str::<Vec<i64>>(key, None, RADIX_TEST_PAGE)
            .iter().map(|s| crate::node::hash_str_to_int64(s)).collect();
        let parent_hash = shared_pages.checked_sub(1).map(|n| hashes[n]);
        assert_eq!(core.take_events(), vec![KvCacheEvent::BlockStored {
            block_hashes: hashes[shared_pages..].to_vec(),
            parent_block_hash: parent_hash,
            token_ids: key[shared_pages * RADIX_TEST_PAGE..].to_vec(),
            block_size: RADIX_TEST_PAGE,
            medium: StorageMedium::Gpu,
            cache_salt: None, session_id: None, extra_key: None,
        }]);
    }
    core.sanity_check(&[], &[]);
}

#[test]
fn radix_page160_full_swa_eviction_reuse() {
    let mut core = UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams {
        page_size: RADIX_TEST_PAGE, swa_sliding_window_size: Some(480), ..Default::default()
    }, vec![FULL, SWA]);
    let physical = radix_page160_slots(&[5, 29, 8]);
    let swa_slots = radix_page160_slots(&[4, 63, 22]);
    for offset in [0i64, 1000, 5000] {
        let key: Vec<i64> = (offset..offset + 480).collect();
        let result = core.insert(&insert_params(&key, &physical));
        let leaf = result.last_device_node_id.unwrap();
        let [action] = result.cache_actions.as_slice() else { panic!("expected one SWA mapping"); };
        let rebuild = expect_action::<crate::cache_action::SwaRebuild>(action.as_ref());
        assert_eq!(rebuild.node_id, leaf);
        assert!(rebuild.source_value.equal(&Tensor::from_slice(&physical)));
        core.set_component_device_value(leaf, SWA, Tensor::from_slice(&swa_slots)).unwrap();
        radix_page160_expect(&mut core, &key, &physical);
        let node = core.arena.resolve(leaf).unwrap();
        assert!(core.arena.device_value(node, SWA).equal(&Tensor::from_slice(&swa_slots)));
        let (_, freed) = core.evict_device_leaf(leaf, false).unwrap();
        assert_eq!(freed.tracker[&FULL], physical.len());
        assert_eq!(freed.tracker[&SWA], physical.len());
        assert!(freed.device_frees[&FULL][0].equal(&Tensor::from_slice(&physical)));
        radix_page160_expect(&mut core, &key, &[]);
        core.sanity_check(&[], &[]);
    }
}

#[test]
fn radix_page160_realistic_swa_window_rounds_out_to_whole_pages() {
    let mut core = UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams {
        page_size: RADIX_TEST_PAGE, swa_sliding_window_size: Some(2048), ..Default::default()
    }, vec![FULL, SWA]);
    let key: Vec<i64> = (0..2560).collect();
    let full = radix_page160_slots(&[31, 9, 27, 14, 22, 3, 89, 7, 45, 4, 1, 52, 29, 17, 67, 15]);
    let swa = radix_page160_slots(&[2, 37, 9, 49, 25, 17, 61, 47, 15, 63, 44, 12, 3, 59, 32, 65]);
    let inserted = core.insert(&insert_params(&key, &full));
    assert_eq!(inserted.cache_actions.len(), 2);
    // 2,048 requested tokens require 13 complete pages (2,080 slots) in the tail.
    let spans = [0..480, 480..2560];
    let mut nodes = Vec::new();
    for (action, span) in inserted.cache_actions.iter().zip(spans.iter()) {
        let rebuild = expect_action::<crate::cache_action::SwaRebuild>(action.as_ref());
        assert!(rebuild.source_value.equal(&Tensor::from_slice(&full[span.clone()])));
        core.set_component_device_value(
            rebuild.node_id, SWA, Tensor::from_slice(&swa[span.clone()])
        ).unwrap();
        nodes.push(rebuild.node_id);
    }
    radix_page160_expect(&mut core, &key, &full);
    assert_eq!(core.get_node_key_lengths(&nodes).unwrap(), [480, 2080]);
    for (node, span) in nodes.iter().zip(spans.iter()).rev() {
        let (_, freed) = core.evict_device_leaf(*node, false).unwrap();
        assert_eq!(freed.tracker[&FULL], span.len());
        assert_eq!(freed.tracker[&SWA], span.len());
        assert!(freed.device_frees[&FULL][0].equal(&Tensor::from_slice(&full[span.clone()])));
        assert!(freed.device_frees[&SWA][0].equal(&Tensor::from_slice(&full[span.clone()])));
    }
    radix_page160_expect(&mut core, &key, &[]);
    core.sanity_check(&[], &[]);
}
