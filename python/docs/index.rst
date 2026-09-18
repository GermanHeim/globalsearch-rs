.. image:: _static/logo.png
   :width: 550px
   :align: center
   :alt: PyGlobalSearch Logo

PyGlobalSearch provides Python bindings for the `globalsearch-rs <https://github.com/GermanHeim/globalsearch-rs>`_ Rust library, implementing the OQNLP algorithm for global optimization.

The OQNLP algorithm combines scatter search metaheuristics with local optimization to effectively find global minima in nonlinear optimization problems.

.. grid:: 2

   .. grid-item-card:: 📖 Book Website
      :link: https://germanheim.github.io/globalsearch-rs-website/

      Practical examples for different types of optimization problems.

   .. grid-item-card:: 🔧 API Reference
      :link: api  
      :link-type: doc

      Complete Python API documentation with all classes, functions, and parameters.

Key Features
------------

- **Multiple Solvers**: COBYLA, L-BFGS, Gradient Descent, Trust Region, Nelder-Mead, L-BFGS-B, Bounded Nelder-Mead, BOBYQA, SLSQP, Barrier, AugmentedLagrangian
- **Constraint Support**: Nonlinear constraints via COBYLA (derivative-free) and SLSQP (gradient-based); linear inequalities via Barrier; linear equalities via AugmentedLagrangian  
- **Gradient Support**: Optional gradient and Hessian functions for faster convergence
- **Flexible Configuration**: Builder pattern for detailed solver customization
- **Performance**: Built in Rust for speed with Python convenience

**Note**: For practical examples and usage patterns, please refer to the `Book Website <https://germanheim.github.io/globalsearch-rs-website/>`_. The API reference here provides detailed documentation of all classes and functions available in PyGlobalSearch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
