from value_based_solver import ValueBasedSolver as VBS
import matplotlib.pyplot as plt
import numpy as np

class ValueBasedSolverReporter:
    def __init__(self, solver, demand_name, criteria_names, experiment_label):
        self._s = solver
        self.d_name = demand_name
        self.c_names = criteria_names
        self._exp = experiment_label

    def plot_min_mean_max(self, criteria, demands, figname, model):
        fig, ax = plt.subplots(int(np.ceil(self._s._m.num_cri / 3))+1, 3, figsize=(12, 9))
        CL = self._s._m.cri_lvls.T * 100
        DL = self._s._m.dem_lvls * 100

        mins, means, maxs = criteria
        Y_min, Y_mean, Y_max = demands

        ax[0, 0].axis('off')
        ax[0, 2].axis('off')
        ax[0, 1].plot(DL, Y_mean, 'g-')
        ax[0, 1].plot(DL, Y_min, 'b--')
        ax[0, 1].plot(DL, Y_max, 'r--')
        ax[0, 1].fill_between(DL, Y_min, Y_max, color='gray', alpha=0.2)
        ax[0, 1].set_title(self.d_name)

        for index in range(3, model.num_cri+3):
            curr_ax = ax[index // 3, index % 3]
            mean_pts = [means[index-3, k] for k in range(model.alpha_i)]
            min_pts = [mins[index-3, k] for k in range(model.alpha_i)]
            max_pts = [maxs[index-3, k] for k in range(model.alpha_i)]
            
            curr_ax.plot(CL[index-3], mean_pts, 'g-')
            curr_ax.plot(CL[index-3], min_pts, 'b--')
            curr_ax.plot(CL[index-3], max_pts, 'r--')
            curr_ax.fill_between(CL[index-3], min_pts, max_pts, color='gray', alpha=0.2)
            curr_ax.set_title(self.c_names[index-3])
        ax[-1, 1].legend(['średnie', 'min', 'max'], loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3, fontsize='x-large' )
        fig.tight_layout()
        fig.savefig('./plots/'+figname)


    def plot_err_bar(self, b_min, b_mean, b_max):
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-whitegrid')
        plt.errorbar(range(1, b_mean.shape[0] + 1), b_mean.tolist(), \
                    yerr=[b_mean - b_min, b_max - b_mean], \
                    fmt='ob', capsize=5, linestyle="None", ecolor='k')
        plt.ylabel('Błąd')
        plt.xticks(np.arange(len(self.c_names) + 1), [''] + self.c_names, rotation=20)
        plt.savefig(f'./plots/{self._exp}_b_postopt.png')

    def plot_post_opt(self):
        b_min, b_mean, b_max = self._s.get_postopt_b()
        __criteria = self._s.get_postopt_x()
        __demands = self._s.get_postopt_y(__criteria[0], __criteria[1], __criteria[2])
        
        self.plot_err_bar(b_min, b_mean, b_max)
        self.plot_min_mean_max(__criteria, __demands, f'{self._exp}_postopt.png', self._s._m)


    def report(self, verbose=False):
        if verbose:
            print("b_i")
            sumka=0
            for x in range(len(self.b)):
                print(x, self.b[x])
                sumka += self.b[x]
            print('Σ =', sumka)
            print()
            print( "x_vals")
            for x in self._s.x_vals:
                print(x, self._s.x_vals[x])
            print()
            print( "y_vals")
            for y in self._s.y_vals:
                print(y, self._s.y_vals[y])

        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-whitegrid')
        timerange = [ str(y) if y//10>0 else f'0{y}' for y in range(self._s._m.time_horizon) ]
        plt.scatter( timerange, self._s.sigm, c=['c'])
        plt.scatter( timerange, self._s.sigp, c=['r'])
        plt.ylabel('Błąd')
        plt.xlabel('Rok')
        plt.legend(['σ -' , 'σ +'], frameon=True)
        plt.savefig(f'./plots/{self._exp}_errors.png')

        fig, ax = plt.subplots(int(np.ceil(self._s._m.num_cri / 3))+1, 3, figsize=(12, 9))
        CL = self._s._m.cri_lvls.T * 100
        DL = self._s._m.dem_lvls * 100

        ax[0, 0].axis('off')
        ax[0, 2].axis('off')
        ax[0, 1].plot(DL, [self._s.y_vals[u] for u in self._s.y_vals])
        ax[0, 1].set_title(self.d_name)

        for index in range(3, self._s._m.num_cri + 3):
            curr_ax = ax[index // 3, index % 3]
            curr_ax.plot(CL[index-3], [self._s.x_vals[index-3, k] for k in range(self._s._m.alpha_i)], 'y-')
            curr_ax.set_title(self.c_names[index-3])

        fig.tight_layout()
        fig.savefig(f'./plots/{self._exp}_plots.png')

    def plot_predictions(self, y_pred):
        train_data = self._s._m.demand
        test_data = self._s._m.demand_test
        plt.figure()
        plt.plot( train_data.index, train_data, 'y-')
        plt.plot( test_data.index, test_data, 'g-')
        plt.plot( test_data[1:].index, y_pred, 'r--')
        #handle incontinuities
        # plt.plot( [train_data.index[-1], test_data.index[0]],
        #         [train_data.iloc[-1], test_data.iloc[0]], 'g-')
        plt.plot( test_data.index[:2],
                [train_data.iloc[-1], y_pred[0]], 'r--')
        plt.legend(['dane uczące', 'dane testowe', 'predykcja'], loc='upper left', frameon=True)
        plt.xlabel('rok')
        plt.ylabel('Zapotrzebowanie [GWH]')
        plt.savefig(f'./plots/{self._exp}_pred_test.png')

    def plot_predictions_and_postopt(self, y_pred, y_min, y_mean, y_max):
        _legend = ['dane uczące', 'dane testowe', 'predykcja', 'min', 'średnie', 'max']
        train_data = self._s._m.demand
        test_data = self._s._m.demand_test[1:]
        train_idx = list(train_data.index)
        pred_idx = list(test_data.index)
        plt.figure()
        plt.plot( train_idx, train_data, 'y-')
        if not test_data.isna().any():
            plt.plot(pred_idx, test_data, 'y--')
        else:
            _legend = [_legend[0]] + _legend[2:]
        plt.plot( pred_idx, y_pred, 'g-')
        plt.plot( pred_idx, y_min, 'c--')
        plt.plot( pred_idx, y_mean, 'b--')
        plt.plot(pred_idx, y_max, 'r--')
        #handle incontinuities
        plt.plot( [train_idx[-1], pred_idx[0]],
                [train_data.iloc[-1], y_pred[0]], 'g--')
        plt.plot( [train_idx[-1], pred_idx[0]],
                [train_data.iloc[-1], y_min[0]], 'c--')
        plt.plot( [train_idx[-1], pred_idx[0]],
                [train_data.iloc[-1], y_mean[0]], 'b--')
        plt.plot( [train_idx[-1], pred_idx[0]],
                [train_data.iloc[-1], y_max[0]], 'r--')
        plt.xlabel('rok')
        plt.ylabel('Zapotrzebowanie [GWH]')
        plt.legend(_legend, loc='best', frameon=True)
        plt.savefig(f'./plots/{self._exp}_predict.png')
