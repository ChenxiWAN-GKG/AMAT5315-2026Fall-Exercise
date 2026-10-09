use seismic::checkpoint::schedule;

#[test]
fn six_steps_use_the_treeverse_split_and_reverse_order() {
    let plan = schedule(6, 2);
    let calls = plan
        .actions
        .iter()
        .filter(|item| item.action == "call")
        .count();
    let grads = plan
        .actions
        .iter()
        .filter(|item| item.action == "grad")
        .map(|item| item.step)
        .collect::<Vec<_>>();
    assert_eq!(calls, 8);
    assert_eq!(grads, vec![5, 4, 3, 2, 1, 0]);
    assert_eq!(plan.peak_saved_states, 3);
    assert_eq!(
        plan.actions
            .iter()
            .find(|item| item.action == "store")
            .unwrap()
            .step,
        3
    );
    assert_eq!(plan.actions.last().unwrap().saved_states, 1);
}

#[test]
fn reflector_work_matches_binomial_schedule() {
    for (budget, expected) in [(1, 28680), (3, 1695), (5, 990), (10, 642)] {
        let plan = schedule(240, budget);
        assert_eq!(plan.forward_calls, expected);
        assert_eq!(plan.peak_saved_states, budget + 1);
        assert_eq!(plan.reverse_calls, 240);
    }
}
