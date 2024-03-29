from bi_threshold.CTL.causal_tree.ctl_trigger.trigger_ctl import *
from sklearn.model_selection import train_test_split


class TriggerBaseNode(TriggerNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class TriggerTreeBase(TriggerTree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = TriggerBaseNode()

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        # ----------------------------------------------------------------
        # Verbosity?
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Split data
        # ----------------------------------------------------------------
        train_x, val_x, train_y, val_y, train_t, val_t = train_test_split(x, y, t, random_state=self.seed, shuffle=True,
                                                                          test_size=self.val_split)
        self.root.num_samples = y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        #effect, trigger = tau_squared_trigger(y, t, self.min_size, self.quartile)
        effect, trigger_d, trigger_u= tau_squared_trigger(y, t, self.min_size, self.quartile)
        #p_val = get_pval_trigger(y, t, trigger)
        p_val = get_pval_trigger(y, t, trigger_d,trigger_u)
        self.root.effect = effect
        self.root.p_val = p_val
        #self.root.trigger = trigger
        self.root.trigger_d = trigger_d
        self.root.trigger_u = trigger_u

        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        #node_eval, trigger, mse = self._eval(train_y, train_t, val_y, val_t)
        node_eval, trigger_d,trigger_u, mse = self._eval(train_y, train_t, val_y, val_t)
        self.root.obj = node_eval

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        #self.root.control_mean = np.mean(y[t >= trigger])
        #self.root.treatment_mean = np.mean(y[t < trigger])
        treat = (t <= trigger_d)& (t >= trigger_u)
        control = ~treat
        self.root.control_mean = np.mean(y[control])
        self.root.treatment_mean = np.mean(y[treat])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, val_x, val_y, val_t)

    def _fit(self, node: TriggerBaseNode, train_x, train_y, train_t, val_x, val_y, val_t):

        if train_x.shape[0] == 0 or val_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect
            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node

        best_gain = 0.0
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)
        #best_tb_trigger, best_fb_trigger = (0.0, 0.0)
        best_tb_trigger_d, best_fb_trigger_d,best_tb_trigger_u, best_fb_trigger_u = (0.0, 0.0, 0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            if self.max_values is not None:
                if self.max_values < 1:
                    idx = np.round(np.linspace(0, len(unique_vals) - 1, self.max_values * len(unique_vals))).astype(int)
                    unique_vals = unique_vals[idx]
                else:
                    idx = np.round(np.linspace(
                        0, len(unique_vals) - 1, self.max_values)).astype(int)
                    unique_vals = unique_vals[idx]

            for value in unique_vals:

                (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                    = divide_set(val_x, val_y, val_t, col, value)

                (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                    = divide_set(train_x, train_y, train_t, col, value)

                #tb_eval, tb_trigger, tb_mse = self._eval(train_y1, train_t1, val_y1, val_t1)
                #fb_eval, fb_trigger, fb_mse = self._eval(train_y2, train_t2, val_y2, val_t2)
                tb_eval, tb_trigger_d,tb_trigger_u, tb_mse = self._eval(train_y1, train_t1, val_y1, val_t1)
                fb_eval, fb_trigger_d,fb_trigger_u, fb_mse = self._eval(train_y2, train_t2, val_y2, val_t2)

                split_eval = (tb_eval + fb_eval)
                gain = -node.obj + split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)
                    #best_tb_trigger, best_fb_trigger = (tb_trigger, fb_trigger)
                    best_tb_trigger_d, best_fb_trigger_d, best_tb_trigger_u, best_fb_trigger_u= (tb_trigger_d, fb_trigger_d,tb_trigger_u, fb_trigger_u)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                = divide_set(val_x, val_y, val_t, node.col, node.value)

            y1 = np.concatenate((train_y1, val_y1))
            y2 = np.concatenate((train_y2, val_y2))
            t1 = np.concatenate((train_t1, val_t1))
            t2 = np.concatenate((train_t2, val_t2))

            #best_tb_effect = ace_trigger(y1, t1, best_tb_trigger)
            #best_fb_effect = ace_trigger(y2, t2, best_fb_trigger)
            #tb_p_val = get_pval_trigger(y1, t1, best_tb_trigger)
            #fb_p_val = get_pval_trigger(y2, t2, best_fb_trigger)
            best_tb_effect = ace_trigger(y1, t1, best_tb_trigger_d, best_tb_trigger_u)
            best_fb_effect = ace_trigger(y2, t2, best_fb_trigger_d, best_fb_trigger_u)

            tb_p_val = get_pval_trigger(y1, t1, best_tb_trigger_d, best_tb_trigger_u)
            fb_p_val = get_pval_trigger(y2, t2, best_fb_trigger_d, best_fb_trigger_u)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            #tb = TriggerBaseNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                   #              node_depth=node.node_depth + 1,
                    #             num_samples=y1.shape[0], trigger=best_tb_trigger)
            #fb = TriggerBaseNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                   #              node_depth=node.node_depth + 1,
                   #              num_samples=y2.shape[0], trigger=best_fb_trigger)
            tb = TriggerBaseNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                                 node_depth=node.node_depth + 1,
                                 num_samples=y1.shape[0], trigger_d=best_tb_trigger_d,trigger_u=best_tb_trigger_u)
            fb = TriggerBaseNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                                 node_depth=node.node_depth + 1,
                                 num_samples=y2.shape[0], trigger_d=best_fb_trigger_d, trigger_u=best_fb_trigger_u)

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, val_x1, val_y1, val_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, val_x2, val_y2, val_t2)

            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect

            return node

        else:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect

            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node
