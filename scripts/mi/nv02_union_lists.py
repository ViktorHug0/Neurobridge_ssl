"""member lists shared by the screens: sub24's ten (cached-output names) and the candidates of rounds 11-12"""
TEN = ['mi_d6_ho5k_lag5_jit3_checkpoint_test_best.pth', 'mi_rmix_ho5k_lag5_jit3_checkpoint_ep10.pth', 'mi_sqfx_ho5k_lag5_jit3_checkpoint_test_best.pth',
       'mi_scr_ho5k_lag5_checkpoint_test_best.pth', 'mi_sqf_ho5k_lag5_checkpoint_ep32.pth', 'mi_plab_ho5k_lag5_jit3_checkpoint_test_best.pth',
       'mi_d6_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth', 'mi_rmix_ho5k_lag5_jit3_soft3_checkpoint_ep10.pth', 'mi_plab_ho5k_lag5_long_soft3_checkpoint_ep10.pth',
       'mi_stf_ho5k_lag5_jit3_soft3_checkpoint_test_best.pth']
CANDS = [f'mi_{r}_checkpoint_test_best.pth' for r in ('stf_ho5k_lag5_jit3', 'prev_ho5k_lag5_jit3_soft3', 'plab4_ho5k_lag5_jit3_soft3', 'cbm_ho5k_lag5_jit3_soft3', 'cbm4_ho5k_lag5_jit3_soft3',
         'stf64_ho5k_lag5_jit3_soft3', 'stf_ho5k_lag5_jit3_soft3s2', 'egp_ho5k_lag5_jit3_soft3', 'luna_ho5k_lag5_jit3_soft3', 'tsc_ho5k_lag5_jit3_soft3', 'rec_ho5k_lag5_jit3_soft3', 'stfscr_ho5k_lag5_jit3_soft3',
         'sqfx_ho5k_lag5_jit3_soft3', 'scr_ho5k_lag5_soft3', 'plab_ho5k_lag5_jit3_soft3', 'cbm_ho5k_lag5', 'prev_ho5k_lag5', 'tsc_ho5k_lag5', 'plab_ho5k_lag5_long')] + \
        ['mi_stfb_ho5k_lag5_jit3_soft3_checkpoint_ep06.pth', 'mi_sqf_ho5k_lag5_soft3_checkpoint_ep32.pth', 'mi_rmix_ho5k_lag5_jit3_tbs_checkpoint_ep10.pth']
