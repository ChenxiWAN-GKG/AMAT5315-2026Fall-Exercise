//! Binomial Treeverse schedule for a fixed number of equal-cost steps.

use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize)]
pub struct Action {
    pub action: &'static str,
    pub step: usize,
    pub saved_states: usize,
}

#[derive(Debug)]
pub struct Schedule {
    pub actions: Vec<Action>,
    pub forward_calls: usize,
    pub reverse_calls: usize,
    pub peak_saved_states: usize,
    saved: BTreeSet<usize>,
}

impl Schedule {
    fn push(&mut self, action: &'static str, step: usize) {
        self.actions.push(Action {
            action,
            step,
            saved_states: self.saved.len(),
        });
    }

    fn recurse(
        &mut self,
        mut delta: usize,
        mut tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
    ) {
        if sigma > beta {
            delta -= 1;
            assert!(self.saved.contains(&beta));
            self.push("restore", beta);
            for step in beta..sigma {
                self.push("call", step);
                self.forward_calls += 1;
            }
            self.saved.insert(sigma);
            self.peak_saved_states = self.peak_saved_states.max(self.saved.len());
            self.push("store", sigma);
        }
        let mut kappa = split(delta, tau, sigma, phi);
        while tau > 0 && kappa < phi {
            self.recurse(delta, tau, sigma, kappa, phi);
            tau -= 1;
            phi = kappa;
            kappa = split(delta, tau, sigma, phi);
        }
        assert!(self.saved.contains(&sigma));
        self.push("grad", sigma);
        self.reverse_calls += 1;
        if sigma > beta {
            self.saved.remove(&sigma);
            self.push("fetch", sigma);
        }
    }
}

fn split(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    if delta + tau == 0 {
        return phi;
    }
    let mut kappa = (delta * sigma + tau * phi + delta + tau - 1) / (delta + tau);
    if kappa >= phi && delta > 0 {
        kappa = (phi - 1).max(sigma + 1);
    }
    kappa
}

fn capacity(delta: usize, tau: usize) -> u128 {
    let mut result = 1_u128;
    for k in 1..=delta {
        result = result.saturating_mul((tau + k) as u128) / (k as u128);
    }
    result
}

pub fn schedule(steps: usize, budget: usize) -> Schedule {
    assert!(steps > 0 && budget > 0);
    let mut tau = 1;
    while capacity(budget, tau) < steps as u128 {
        tau += 1;
    }
    let mut plan = Schedule {
        actions: Vec::new(),
        forward_calls: 0,
        reverse_calls: 0,
        peak_saved_states: 1,
        saved: BTreeSet::from([0]),
    };
    plan.recurse(budget, tau, 0, 0, steps);
    assert_eq!(plan.saved, BTreeSet::from([0]));
    plan
}
