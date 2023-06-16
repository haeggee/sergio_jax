from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import jit


class SergioJAX:
    """ JAX implementation of SERGIO, a simulator for creating realistic single-cell expression data.
        The simulator gives the expression of genes based on a gene regulatory network (GRN) using 
        the Chemical Langevin Equation (CLE). This implementation focuses on the steady-state simulation of gene expressions.
        
        The simulation is mostly governed by the parameters:
        * graph: adjacency matrix of the GRN
        * contribution_rates: contribution rates of the regulators, i.e. the weights of the edges in the GRN
        * basal_rates: basal rates of the genes
        * hill: hill coefficients

        These parameters can be fixed via the custom_graph function or adapted to real data (TODO Yunshu).

        Additional parameters include the decay rates of the genes and the noise amplitude as well as the type of noise.

        The resulting SDE can be written as:
            
            x_(t+1) = x_t + (P_t - decay * x_(t)) dt + \
                noise_amp * sqrt{P_(t)} dW_alpha + noise_amp sqrt{decay * x_(t)}  dW_beta
        
        where x_t is the gene expression at time t, dW_beta and dW_alpha are independent Wiener processes (Brownian motions)
        as dW_alpha = sqrt{dt} * N(0, 1) and dW_beta = sqrt{dt} * N(0, 1).
        P_t is the production rate at time t. The production rate for each gene is determined by its regulators.


        Please refer to the original paper for more details:
            Dibaeinia, P., & Sinha, S. (2020). SERGIO: a single-cell expression simulator guided by gene regulatory networks. Cell systems, 11(3), 252-271.
        
        Link: 
            https://www.sciencedirect.com/science/article/pii/S2405471220302878

        GitHub:
            https://github.com/PayamDiba/SERGIO

    Args:
        n_genes (int): number of genes in the GRN
        n_cell_types (int): number of cell types in the GRN
        n_sc (int): number of single cells to simulate (per cell type)
        noise_amplitude (float or jax.Array): amplitude of the noise in CLE. This can be a scalar to use for all genes or an array with the same size as n_genes.
        noise_type (str): type of noise in CLE. 
            We consider three types of noise, 
                * 'sp': a single intrinsic noise is associated to production process
                * 'spd': a single intrinsic noise is associated to both production and decay processes
                * 'dpd': two independent intrinsic noises are associated to production and decay processes
        decays (float or jax.Array): decay rates of the genes. This can be a scalar to use for all genes or an array with the same size as n_genes.
        sampling_state: number of steady-state steps to simulate for sampling.
            single cells are sampled from sampling_state * number_sc steady-state steps
        init_steps: number of steps to run a fixed point iteration for finding
            the steady-state of the system, i.e. initial gene expressions / production rates
        dt (float): time step used in CLE
        safety_steps (int): number of steps to additionally run the simulation (as a safety check)
            which are to be ignored for actual sampling of the expressions
    """

    def __init__(self,
                 n_genes: int,
                 n_cell_types: int,
                 n_sc: int,
                 noise_amplitude: Union[jax.Array, float] = 1.0,
                 noise_type: str = 'dpd',
                 decays: Union[jax.Array, float] = 0.8,
                 sampling_state: int = 10,
                 init_steps: int = 10,
                 dt: float = 0.01,
                 safety_steps: int = 0):

        assert noise_type in [
            'sp', 'spd', 'dpd'
        ], 'noise_type should be one of the following: sp, spd, dpd'
        assert jnp.isscalar(
            noise_amplitude) or noise_amplitude.shape == (  # type: ignore
                n_genes,
            ), 'noise_amplitude should be a scalar or a vector of size n_genes'
        assert jnp.isscalar(decays) or decays.shape == (  # type: ignore
            n_genes, ), 'decays should be a scalar or a vector of size n_genes'

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.number_sc = n_sc
        self.sampling_state = sampling_state
        self.dt = dt
        self.noise_type = noise_type

        self.noise_amplitude = noise_amplitude
        self.decays = decays

        self.init_steps = init_steps
        self.safety_steps = safety_steps

    def custom_graph(
        self,
        graph: jax.Array,
        contribution_rates: jax.Array,
        basal_rates: jax.Array,
        hill: Union[jax.Array, float] = 2.0,
    ):
        """ Insert custom graph for the gene regulatory network (GRN). 
            This function is used to initialize the simulator with a custom GRN.
            
        Args:
            graph (jax.Array): [n_genes, n_genes] 
                adjacency matrix of the GRN
            contribution_rates (jax.Array): [n_cell_types, n_genes, n_genes]
                contribution rates of the regulators
            basal_rates (jax.Array): [n_cell_types, n_genes] basal rates of the target genes
            hill (Union[jax.Array, float], optional): hill coefficient. Defaults to 2.0.
        """
        assert graph.shape == (self.n_genes, self.n_genes), \
                'graph should be of size [n_genes, n_genes]'
        assert contribution_rates.shape == (self.n_cell_types, self.n_genes, self.n_genes), \
            'contribution_rates should be of size [n_cell_types, n_genes, n_genes]'
        assert basal_rates.shape == (self.n_cell_types, self.n_genes), \
                'basal_rates should be of size [n_genes, ]'
        assert jnp.isscalar(hill) or hill.shape == (  # type: ignore
            self.n_genes,
        ), 'hill should be a scalar or a vector of size n_genes'

        self.graph = graph
        # repeat for different cell types
        self.contribution_rates = contribution_rates
        self.basal_rates = basal_rates
        # make sure to zero out basal rates for non-source genes (no master-regulators)
        self.basal_rates *= (graph.sum(axis=0) == 0)
        self.hill = hill

    def hill_function(self,
                      x: jax.Array,
                      h: jax.Array,
                      n: Union[jax.Array, float] = 2.0) -> jax.Array:
        """ Hill function for computing the contribution of a 
            regulator to the expression of a target gene.

        Args:
            x (jax.Array): input value
            k (jax.Array): half response value
            n (jax.Array): hill coefficient (i.e. the power)
        
        Returns:
            jax.Array: output value
        """

        hill_numerator = jnp.power(x, n)
        hill_denominator = jnp.power(h, n) + hill_numerator
        hill_denominator = jnp.where(
            jnp.abs(hill_denominator) < 1e-6, 1e-6, hill_denominator)
        return hill_numerator / (hill_denominator)

    def production_rate(self,
                        gene_exp: jax.Array,
                        graph: jax.Array,
                        basal_rates: jax.Array,
                        contribution_rates: jax.Array,
                        hill: Union[jax.Array, float] = 2.0) -> jax.Array:
        """ Calculate the production rate of each gene in each cell type
        
        Args:
            gene_exp (jax.Array): [n_cell_types, n_genes] gene expression
            graph (jax.Array): [n_genes, n_genes] graph adjacency matrix,
              i.e. graph[i,j] = 1 if i is a regulator for j, i->j
            basal_rates (jax.Array): [n_cell_types, n_genes] basal rates 
                for master regulators (source nodes), i.e. b_i
            contribution_rates (jax.Array): [n_cell_types, n_genes, n_genes] 
                contribution rates / interaction coefficients, i.e. K_ij in the original paper
            hill (float): hill coefficient for the hill function. the same value is used for all gene combinations
                TODO: have hill matrix (more flexible simulation) -- currently, code just works for hill as constant float
        Returns:
            production_rate (jax.Array): [n_cell_types, n_genes] production rate
        """
        # [n_genes,]
        half_response = gene_exp.mean(0)

        # [n_cell_types, n_genes]
        hills = self.hill_function(gene_exp, half_response[None], hill)
        # [n_cell_types, n_genes, 1]
        hills = hills[:, :, None]
        # [n_cell_types, n_genes, n_genes]
        masked_contribution = graph[None] * contribution_rates

        # [n_cell_types, n_genes, n_genes]
        # switching mechanism between activation and repression,
        # which is decided via the sign of the contribution rates
        intermediate = jnp.where(masked_contribution > 0,
                                 jnp.abs(masked_contribution) * hills,
                                 jnp.abs(masked_contribution) * (1 - hills))
        # [n_cell_types, n_genes]
        # sum over regulators, i.e. sum over i in [b, i, j]
        production_rate = basal_rates + intermediate.sum(1)
        return production_rate

    @partial(jit, static_argnums=(0, 7))
    def init_gene_concentration(self,
                                graph: jax.Array,
                                basal_rates: jax.Array,
                                contribution_rates: jax.Array,
                                hill: Union[jax.Array, float],
                                decay: Union[jax.Array, float],
                                kout: Optional[jax.Array] = None,
                                steps: int = 10) -> jax.Array:
        """ Estimate the steady-state gene concentration in all cell types,
            used for initializing the simulation.
            
            For master-regulators (sources), the steady-state concentration
            is simply the basal rate / decay (i.e. b_i / lambda_i). 
            For non-master regulators, the steady-state concentration is estimated
            by iterating the production rate function until convergence
            (or fixed number of steps), divided by the decay rate, i.e.

                E[x_i] = sum(p_ij ( E[x_j] )) / lambda_i

            where p_ij() is the production rate function.
            One can see that this is a recursive relation (which stops at master regulators),
            which we solve via fixed-point iteration.
            

        Args:
            graph (jax.Array): [n_genes, n_genes] graph adjacency matrix,
              i.e. graph[i,j] = 1 if i is a regulator for j, i->j
            basal_rates (jax.Array): [n_cell_types, n_genes] basal rates, i.e. b_i
            contribution_rates (jax.Array): [n_cell_types, n_genes, n_genes] 
                contribution rates / interaction coefficients, i.e. K_ij in the original paper
            hill (float): hill coefficient for the hill function. the same value is used for all gene combinations
            decay (float): decay rate for all genes
            kout (jax.Array, optional): [n_genes,] knockout rates for all genes.
                1.0 corresponds to full knockout, 0.5 to a "half" knockdown. Defaults to None.
            steps (int): number of steps to iterate 

        Returns:
            gene_exp (jax.Array): [n_cell_types, n_genes] gene expression
        """
        if kout is None:
            # [n_genes,]
            kout = jnp.zeros(graph.shape[0])

        init_concs = jnp.copy(basal_rates) * (1 - kout)

        def update(_, gene_exp):
            production_rate = self.production_rate(gene_exp, graph,
                                                   basal_rates,
                                                   contribution_rates, hill)
            return (production_rate * (1 - kout)) / decay

        gene_exp = jax.lax.fori_loop(0, steps, update, init_concs)
        return gene_exp

    def _simulation_step(self,
                         rng: jax.random.PRNGKeyArray,
                         gene_exp: jax.Array,
                         graph: jax.Array,
                         basal_rates: jax.Array,
                         contribution_rates: jax.Array,
                         hill: Union[jax.Array, float],
                         decay: Union[jax.Array, float],
                         kout: Optional[jax.Array] = None,
                         noise_type: str = "dpd",
                         noise_amp: float = 1.0,
                         dt: float = 0.01) -> jax.Array:
        """ One simulation step of the gene expression model.
            The simulation is done in parallel for all cell types.
            
            The formula for the simulation step is:
            
            x_(t+1) = x_t + (P_t - decay * x_(t)) dt + \
                noise_amp * sqrt{P_(t)} dW_alpha + noise_amp sqrt{decay * x_(t)}  dW_beta

            where dW_beta and dW_alpha are independent Wiener processes (Brownian motions)
            as dW_alpha = sqrt{dt} * N(0, 1) and dW_beta = sqrt{dt} * N(0, 1).
            P_t is the production rate at time t.
            
        Args:
            gene_exp (jax.Array): [n_cell_types, n_genes] gene expression
            graph (jax.Array): [n_genes, n_genes] graph adjacency matrix,
              i.e. graph[i,j] = 1 if i is a regulator for j, i->j
            basal_rates (jax.Array): [n_cell_types, n_genes] basal rates, i.e. b_i
            contribution_rates (jax.Array): [n_cell_types, n_genes, n_genes]
                contribution rates / interaction coefficients, i.e. K_ij in the original paper
            hill (float): hill coefficient for the hill function. 
                the same value is used for all gene combinations
            decay (float): decay rate for all genes
            noise_type (str): type of noise to add to the simulation
                "sp" for single production (beta=0), "spd" for single decay noise (alpha=0), 
                "dpd" for double noise
            noise_amp (float): amplitude of the noise
            dt (float): time step for the simulation

        Returns:
            gene_exp (jax.Array): [n_cell_types, n_genes] updated gene expression
        """
        if kout is None:
            # [n_genes,]
            kout = jnp.zeros(graph.shape[0])

        # gene_exp = gene_exp * (1 - kout)
        production_rate = self.production_rate(gene_exp, graph, basal_rates,
                                               contribution_rates, hill)
        production_rate = production_rate * (1 - kout)
        decay = gene_exp * decay
        rng, rng_dw1, rng_dw2 = jax.random.split(rng, 3)

        dw = jax.random.normal(rng_dw1, shape=gene_exp.shape)
        if noise_type == "sp":
            noise = dw * jnp.sqrt(production_rate)
        elif noise_type == "spd":
            noise = dw * jnp.sqrt(decay)
        elif noise_type == "dpd":
            dw2 = jax.random.normal(rng_dw2, shape=gene_exp.shape)
            amplitude_a = jnp.sqrt(production_rate)
            amplitude_b = jnp.sqrt(decay)
            noise = dw * amplitude_a + dw2 * amplitude_b
        else:
            raise ValueError("noise_type must be one of 'sp', 'spd', 'dpd'")

        new_gene_exp = gene_exp + dt * (production_rate - decay) + (
            noise_amp * noise * jnp.sqrt(dt))
        return jnp.clip(new_gene_exp, a_min=0.0, a_max=None)

    @partial(jit, static_argnums=(0, 7, 10, 11))
    def _simulate(self,
                  rng: jax.random.PRNGKeyArray,
                  graph: jax.Array,
                  basal_rates: jax.Array,
                  contribution_rates: jax.Array,
                  hill: Union[jax.Array, float],
                  decay: Union[jax.Array, float],
                  noise_type: str = "dpd",
                  noise_amp: float = 1.0,
                  kout: Optional[jax.Array] = None,
                  steps: int = 1000,
                  init_steps: int = 10,
                  dt: float = 0.01):
        """ Simulate the gene expression model (the stochastic differential equations)
            for a given number of steps.
        
        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            graph (jax.Array): [n_genes, n_genes] graph adjacency matrix,
              i.e. graph[i,j] = 1 if i is a regulator for j, i->j
            basal_rates (jax.Array): [n_cell_types, n_genes] basal rates, i.e. b_i
            contribution_rates (jax.Array): [n_cell_types, n_genes, n_genes]
                contribution rates / interaction coefficients, i.e. K_ij in the original paper
            hill (float): hill coefficient for the hill function. 
                the same value is used for all gene combinations
            decay (float): decay rate for all genes
            noise_type (str): type of noise to add to the simulation
                "sp" for single production (beta=0), "spd" for single decay noise (alpha=0), 
                "dpd" for double noise
            noise_amp (float): amplitude of the noise
            steps (int): number of simulation steps
            init_steps (int): number of initial steps to for the simulation to reach a steady state
            dt (float): time step for the simulation

        Returns:
            gene_exp (jax.Array): [steps, n_cell_types, n_genes] #steps times a snapshot of gene expressions
        """
        if kout is None:
            # [n_genes,]
            kout = jnp.zeros(graph.shape[0])

        def step(gene_exp, rng):
            new_gene_exp = self._simulation_step(
                gene_exp=gene_exp,
                graph=graph,
                basal_rates=basal_rates,
                contribution_rates=contribution_rates,
                hill=hill,
                decay=decay,
                noise_type=noise_type,
                noise_amp=noise_amp,
                kout=kout,
                dt=dt,
                rng=rng)
            return new_gene_exp, new_gene_exp

        init_exp = self.init_gene_concentration(
            graph=graph,
            basal_rates=basal_rates,
            contribution_rates=contribution_rates,
            hill=hill,
            decay=decay,
            kout=kout,
            steps=init_steps)

        rng, *rngs = jax.random.split(rng, steps + 1)
        final_exp, gene_exps = jax.lax.scan(step,
                                            init_exp,
                                            jnp.array(rngs),
                                            length=steps)
        # [n_steps, n_cell_types, n_genes]
        return gene_exps

    def convert_to_UMIcounts(self, rng: jax.random.PRNGKeyArray,
                             scData: jax.Array):
        """ Convert the gene expression to UMI counts.
            This is done by sampling from a poisson distribution with the mean
            of the gene expression.

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            scData (jax.Array): gene expression in float form, any shape
        
        Returns:
            scData (jax.Array): gene expression in UMI counts
        """
        return jax.random.poisson(rng, scData)

    def simulate(self,
                 rng: jax.random.PRNGKeyArray,
                 kout: Optional[jax.Array] = None):
        """ Simulate the gene expression model with the given parameters in the class.

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            kout (jax.Array, optional): knockout mask indicator. Defaults to None.
        """
        if kout is None:
            kout = jnp.zeros(self.n_genes)
        # [n_steps, n_cell_types, n_genes]
        self.gene_exp = self._simulate(
            rng,
            graph=self.graph,
            basal_rates=self.basal_rates,
            contribution_rates=self.contribution_rates,
            hill=self.hill,
            decay=self.decays,
            noise_type=self.noise_type,
            noise_amp=self.noise_amplitude,
            kout=kout,
            steps=self.sampling_state * self.number_sc + self.safety_steps,
            init_steps=self.init_steps,
            dt=self.dt,
        )

    def getExpressions(self, rng: jax.random.PRNGKeyArray):
        """ Get the simulated gene expressions after calling the simulate method.

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
        
        Returns:
            gene_exp (jax.Array): [n_cell_types, n_genes, n_cells_per_type] simulated gene expressions
        """

        steps = self.gene_exp.shape[0]

        rng, sample_rng = jax.random.split(rng)
        sc_indices = jax.random.randint(
            sample_rng,
            shape=(self.number_sc, ),
            maxval=steps,
            # don't use the first #safety_steps no. of steps
            minval=self.safety_steps,
        )
        # [n_cell_types, n_genes, n_cells_per_type]
        return self.gene_exp[sc_indices].transpose((1, 2, 0))

    """""" """""" """""" """""" """""" """""" """
    "" This part is to add technical noise
    """ """""" """""" """""" """""" """""" """"""

    @staticmethod
    @jit
    def outlier_effect(rng: jax.random.PRNGKeyArray,
                       scData: jax.Array,
                       outlier_prob: float = 0.01,
                       mean: float = 0.,
                       scale: float = 1.0):
        """ Add outlier effect to the simulated data.

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            scData (jax.Array): [n_cell_types, n_genes, n_cells_per_type] simulated data
            outlier_prob (float): probability of a gene to be an outlier
            mean (float): mean of the lognormal distribution for outlier factors
            scale (float): scale of the lognormal distribution for outlier factors

        Returns:
            scData (jax.Array): [n_cell_types, n_genes, n_cells_per_type] simulated data with outliers
        """
        n_genes = scData.shape[1]
        rng, sample_rng = jax.random.split(rng)
        out_indicator = jax.random.bernoulli(sample_rng,
                                             p=outlier_prob,
                                             shape=(n_genes, ))

        #### generate outlier factors ####
        rng, sample_rng = jax.random.split(rng)
        outFactors = jax.random.normal(sample_rng,
                                       shape=(n_genes, )) * scale + mean
        # lognormal is the exp of normal samples
        outFactors = jnp.exp(outFactors)
        ##################################

        return jnp.where(
            out_indicator[None, :, None],
            scData * outFactors[None, :, None],
            scData,
        )

    @staticmethod
    @jit
    def lib_size_effect(rng: jax.random.PRNGKeyArray,
                        scData: jax.Array,
                        mean: float = 0.,
                        scale: float = 1.0):
        """
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            scData (jax.Array): the simulated data representing mRNA levels (concentrations);
                                shape [n_cell_types, n_genes, n_cells_per_type]
            mean: mean for log-normal distribution
            var: var for log-normal distribution

        Returns:
            libFactors (jax.Array): the factors used to adjust mRNA levels; shape [n_cell_types, n_cells_per_type]
            modified single cell data (jax.Array): [n_cell_types, n_genes, n_cells_per_type]
        """

        n_cell_types, n_genes, number_sc = scData.shape
        ret_data = []
        rng, sample_rng = jax.random.split(rng)
        libFactors = jax.random.normal(
            sample_rng, shape=(n_cell_types, number_sc)) * scale + mean
        # lognormal is the exp of normal samples
        libFactors = jnp.exp(libFactors)
        # TODO (alex): get rid of for loop, write in proper jax
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = jnp.sum(binExprMatrix, axis=0)
            binFactors = binFactors / jnp.where(normalizFactors == 0.0, 1.0,
                                                normalizFactors)
            binFactors = binFactors.reshape(1, number_sc)
            binFactors = jnp.repeat(binFactors, n_genes, axis=0)

            ret_data.append(binExprMatrix * binFactors)

        return libFactors, jnp.array(ret_data)

    @staticmethod
    @jit
    def dropout_indicator(rng: jax.random.PRNGKeyArray,
                          scData: jax.Array,
                          shape: int = 1,
                          percentile: int = 65):
        """ This function generates binary indicators for dropouts. 
            (Note by original SERGIO authors: This is similar to the Splat package)

        Args:
            rng (jax.random.PRNGKeyArray): random number generator key
            scData (jax.Array): the simulated data representing mRNA levels (concentrations);
                can be the output of the simulator or any refined version of it
                (e.g. with technical noise)
            shape (int): the shape of the logistic function
            percentile (int): the mid-point of logistic functions is set to the given percentile
                of the input scData

        Returns: 
            binary_ind (jax.Array): binary indicators for dropouts; shape
             [n_cell_types, n_genes, n_cells_per_type]
        """

        scData_log = jnp.log(scData + 1)
        log_mid_point = jnp.percentile(scData_log, percentile)
        prob_ber = jnp.true_divide(
            1, 1 + jnp.exp(-1 * shape * (scData_log - log_mid_point)))

        rng, sample_rng = jax.random.split(rng)
        binary_ind = jax.random.bernoulli(sample_rng, p=prob_ber)

        return binary_ind
