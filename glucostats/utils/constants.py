available_statistics = {
    'time_stats': {
        'time_in_ranges': ['t_ir', 't_ar', 't_br', 't_or'],
        'percentage_time_in_ranges': ['pt_ir', 'pt_ar', 'pt_br', 'pt_or']
    },
    'observations_stats': {
        'observations_in_ranges': ['n_ir', 'n_ar', 'n_br', 'n_or'],
        'percentage_observations_in_ranges': ['pn_ir', 'pn_ar', 'pn_br', 'pn_or']
    },
    'descriptive_stats': {
        'mean_in_ranges': ['mean_ir', 'mean_ar', 'mean_br', 'mean_or'],
        'distribution': ['max', 'min', 'max_diff', 'mean', 'std', 'quartiles', 'iqr'],
        'complexity': ['entropy', 'dfa'],
        'auc': ['auc'],
    },
    'risks_stats': {
        'g_indexes': ['lbgi', 'max_lbgi', 'hbgi', 'max_hbgi', 'bgri'],
        'g_risks': ['vlow', 'low', 'high', 'vhigh', 'gri'],
        'grade_stats': ['grade', 'grade_hypo', 'grade_eu', 'grade_hyper'],
    },
    'control_stats': {
        'control_indexes': ['hyper_index', 'hypo_index', 'igc'],
        'a1c': ['eA1C', 'gmi'],
        'qgc': ['m_value', 'j_index'],
    },
    'variability_stats': {
        'excursions': ['mage', 'ef'],
        'variability': ['dt', 'mag', 'gvp', 'cv']
    }
}

groups = ['time_stats',
          'observations_stats',
          'descriptive_stats',
          'risks_stats',
          'control_stats',
          'variability_stats']

subgroups = ['time_in_ranges', 'percentage_time_in_ranges',
             'observations_in_ranges', 'percentage_observations_in_ranges',
             'mean_in_ranges', 'distribution', 'complexity', 'auc',
             'g_indexes', 'g_risks', 'grade_stats',
             'control_indexes', 'a1c', 'qgc',
             'excursions', 'variability']

statistics = ['t_ir', 't_ar', 't_br', 't_or', 'pt_ir', 'pt_ar', 'pt_br', 'pt_or',
              'n_ir', 'n_ar', 'n_br', 'n_or', 'pn_ir', 'pn_ar', 'pn_br', 'pn_or',
              'mean_ir', 'mean_ar', 'mean_br', 'mean_or', 'max', 'min', 'max_diff', 'mean', 'std', 'quartiles', 'iqr', 'entropy', 'dfa', 'auc',
              'lbgi', 'max_lbgi', 'hbgi', 'max_hbgi', 'bgri', 'vlow', 'low', 'high', 'vhigh', 'gri', 'grade', 'grade_hypo', 'grade_eu', 'grade_hyper',
              'hyper_index', 'hypo_index', 'igc', 'eA1C', 'gmi', 'm_value', 'j_index',
              'mage', 'ef', 'dt', 'mag', 'gvp', 'cv']

possible_names = groups + subgroups + statistics

windowing_methods = ['number', 'static', 'dynamic', 'personalized']
