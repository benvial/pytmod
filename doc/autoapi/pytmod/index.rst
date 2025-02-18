pytmod
======

.. py:module:: pytmod

.. autoapi-nested-parse::

   
   This module implements the pytmod API.
















   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   pytmod.Material
   pytmod.Slab


Package Contents
----------------

.. py:class:: Material(eps_fourier, modulation_frequency, Npad=0)

   
   Material object


   :Parameters:

       **eps_fourier** : array_like
           The Fourier coefficients of the dielectric function

       **modulation_frequency** : float
           The modulation frequency of the dielectric function

       **Npad** : int, optional
           The number of components to pad the dielectric function with







   :Raises:

       ValueError
           If the length of `eps_fourier` is even







   ..
       !! processed by numpydoc !!

   .. py:attribute:: modulation_frequency


   .. py:method:: pad(x)

      
      Pad an array with zeros if `Npad` is positive


      :Parameters:

          **x** : array_like
              The array to pad



      :Returns:

          **y** : array_like
              The padded array











      ..
          !! processed by numpydoc !!


   .. py:property:: eps_fourier

      
      The Fourier coefficients of the dielectric function





      :Returns:

          **eps_fourier** : array_like
              The Fourier coefficients of the dielectric function











      ..
          !! processed by numpydoc !!


   .. py:property:: Npad

      
      The number of zeros to pad the Fourier coefficients with





      :Returns:

          **Npad** : int
              The number of zeros to pad the Fourier coefficients with











      ..
          !! processed by numpydoc !!


   .. py:property:: modulation_period

      
      The modulation period of the dielectric function





      :Returns:

          **modulation_period** : float
              The modulation period of the dielectric function











      ..
          !! processed by numpydoc !!


   .. py:property:: nh

      
      The length of the Fourier coefficients array





      :Returns:

          **nh** : int
              The length of the Fourier coefficients array











      ..
          !! processed by numpydoc !!


   .. py:property:: Nh

      
      The integer corresponding to order 0 in the Fourier coefficients array





      :Returns:

          **Nh** : int
              The integer corresponding to order 0 in the Fourier coefficients array











      ..
          !! processed by numpydoc !!


   .. py:method:: index_shift(i)

      
      Shift an index to the index of the Fourier coefficient of the same order
      in the padded array.


      :Parameters:

          **i** : int
              The index in the unpadded array



      :Returns:

          int
              The corresponding index in the padded array











      ..
          !! processed by numpydoc !!


   .. py:method:: build_matrix(omegas)

      
      Build the matrix of the linear system to be solved.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.



      :Returns:

          **matrix** : array_like
              The matrix of the linear system.











      ..
          !! processed by numpydoc !!


   .. py:method:: build_dmatrix_domega(omegas)

      
      Build the matrix derivative wrt omega of the linear system to be solved.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.



      :Returns:

          **dmatrix** : array_like
              The matrix matrix derivative wrt omega.











      ..
          !! processed by numpydoc !!


   .. py:method:: eigensolve(omegas, matrix=None, left=False, normalize=False, sort=True)

      
      Solve the eigenvalue problem for the material.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.

          **matrix** : array_like, optional
              The matrix of the linear system. If None, it will be built.

          **left** : bool, optional
              Whether to compute the left eigenvectors. Defaults to False.

          **normalize** : bool, optional
              Whether to normalize the left and right eigenvectors. Defaults to False.

          **sort** : bool, optional
              Whether to sort the eigenvalues. Defaults to True.



      :Returns:

          **eigenvalues** : array_like
              The eigenvalues of the material.

          **modes_right** : array_like
              The right eigenvectors of the material.

          **modes_left** : array_like
              The left eigenvectors of the material, if left is True.











      ..
          !! processed by numpydoc !!


   .. py:method:: get_modes_normalization(modes_right, modes_left)

      
      Compute the normalization constants for the modes.


      :Parameters:

          **modes_right** : array_like
              The right eigenvectors of the material.

          **modes_left** : array_like
              The left eigenvectors of the material.



      :Returns:

          **normas** : array_like
              The normalization constants.











      ..
          !! processed by numpydoc !!


   .. py:method:: normalize(modes_right, modes_left)

      
      Normalize the eigenmodes of the material.


      :Parameters:

          **modes_right** : array_like
              The right eigenvectors of the material.

          **modes_left** : array_like
              The left eigenvectors of the material.



      :Returns:

          **modes_right** : array_like
              The normalized right eigenvectors of the material.

          **modes_left** : array_like
              The normalized left eigenvectors of the material.








      .. rubric:: Notes

      First, the eigenmodes are normalized so that the left and right
      eigenmodes are biorthogonal. Then, the right eigenmodes are
      normalized so that the maximum value of each eigenmode is 1.



      ..
          !! processed by numpydoc !!


   .. py:method:: get_deigenvalues_domega(omegas, eigenvalues, normalized_modes_right, normalized_modes_left, dmatrix=None)

      
      Compute the derivative of the eigenvalues wrt omega.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to compute the derivative.

          **eigenvalues** : array_like
              The eigenvalues of the material.

          **normalized_modes_right** : array_like
              The normalized right eigenvectors of the material.

          **normalized_modes_left** : array_like
              The normalized left eigenvectors of the material.

          **dmatrix** : array_like, optional
              The matrix derivative wrt omega. Defaults to None.



      :Returns:

          **deigenvalues** : array_like
              The derivative of the eigenvalues wrt omega.











      ..
          !! processed by numpydoc !!


   .. py:method:: get_deigenmodes_right_domega(omegas, eigenvalues, normalized_modes_right, normalized_modes_left, dmatrix=None)

      
      Compute the derivative of the right eigenmodes wrt omega.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to compute the derivative.

          **eigenvalues** : array_like
              The eigenvalues of the material.

          **normalized_modes_right** : array_like
              The normalized right eigenvectors of the material.

          **normalized_modes_left** : array_like
              The normalized left eigenvectors of the material.

          **dmatrix** : array_like, optional
              The matrix derivative wrt omega. Defaults to None.



      :Returns:

          **deigenmodes_right** : array_like
              The derivative of the right eigenmodes wrt omega.











      ..
          !! processed by numpydoc !!


   .. py:method:: freq2time(coeff, t)

      
      Compute the time-domain representation of a coefficient array.


      :Parameters:

          **coeff** : array_like
              The coefficient array in the frequency domain

          **t** : array_like
              The time array at which to compute the time-domain representation



      :Returns:

          array_like
              The time-domain representation of the coefficient array











      ..
          !! processed by numpydoc !!


   .. py:method:: get_eps_time(t)

      
      Compute the time-domain representation of the dielectric function.


      :Parameters:

          **t** : array_like
              The time array at which to compute the time-domain representation



      :Returns:

          array_like
              The time-domain representation of the dielectric function











      ..
          !! processed by numpydoc !!


.. py:class:: Slab(material, thickness, eps_plus=1, eps_minus=1)

   
   Slab object


   :Parameters:

       **material** : Material
           The material of the slab

       **thickness** : float
           The thickness of the slab

       **eps_plus** : float, optional
           The permittivity of the medium above the slab

       **eps_minus** : float, optional
           The permittivity of the medium below the slab














   ..
       !! processed by numpydoc !!

   .. py:attribute:: material


   .. py:attribute:: thickness


   .. py:attribute:: eps_plus
      :value: 1



   .. py:attribute:: eps_minus
      :value: 1



   .. py:attribute:: dim


   .. py:method:: build_matrix(omegas, eigenvalues, modes)

      
      Build the matrix of the linear system to be solved.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.

          **eigenvalues** : array_like
              The eigenvalues of the material.

          **modes** : array_like
              The eigenvectors of the material.



      :Returns:

          **matrix_slab** : array_like
              The matrix of the linear system.











      ..
          !! processed by numpydoc !!


   .. py:method:: build_dmatrix_domega(omegas, eigenvalues, modes, modes_left)

      
      Build the of the linear system to be solved.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.

          **eigenvalues** : array_like
              The eigenvalues of the material.

          **modes** : array_like
              The eigenvectors of the material.

          **modes_left** : array_like
              The left eigenvectors of the material.



      :Returns:

          **matrix_slab** : array_like
              The matrix derivative wrt omega of the linear system.











      ..
          !! processed by numpydoc !!


   .. py:method:: build_rhs(omegas, Eis)

      
      Build the right-hand side (RHS) of the linear system for the slab.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to solve the system.

          **Eis** : array_like
              The incident electric fields.



      :Returns:

          **rhs_slab** : array_like
              The RHS matrix of the linear system.











      ..
          !! processed by numpydoc !!


   .. py:method:: solve(matrix_slab, rhs_slab)

      
      Solve the linear system defined by the matrix and RHS of the slab.


      :Parameters:

          **matrix_slab** : array_like
              The matrix of the linear system.

          **rhs_slab** : array_like
              The right-hand side of the linear system.



      :Returns:

          **solution** : array_like
              The solution of the linear system.











      ..
          !! processed by numpydoc !!


   .. py:method:: extract_coefficients(solution, Eis, kns, ens)

      
      Extracts the coefficients of the waves from the solution of the linear system.


      :Parameters:

          **solution** : array_like
              The solution of the linear system.

          **Eis** : array_like
              The incident electric fields.

          **kns** : array_like
              The eigenvalues of the slab time-modulated medium.

          **ens** : array_like
              The eigenvectors of the slab time-modulated medium.



      :Returns:

          **Eslab_plus** : array_like
              The coefficients of the forward propagating waves inside the slab.

          **Eslab_minus** : array_like
              The coefficients of the backward propagating waves inside the slab.

          **Er** : array_like
              The coefficients of the reflected waves.

          **Et** : array_like
              The coefficients of the transmitted waves.











      ..
          !! processed by numpydoc !!


   .. py:method:: fresnel_static(omegas)

      
      Compute the Fresnel coefficients for a static slab with the same thickness
      and dielectric properties as the current slab.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to compute the Fresnel coefficients



      :Returns:

          **rf** : array_like
              The reflection Fresnel coefficient

          **tf** : array_like
              The transmission Fresnel coefficient











      ..
          !! processed by numpydoc !!


   .. py:method:: eigenvalue_static(n)

      
      Calculate the static eigenvalue for a given mode number.


      :Parameters:

          **n** : int
              The mode number for which the static eigenvalue is calculated.



      :Returns:

          complex
              The static eigenvalue corresponding to the specified mode number,
              based on the slab's thickness and dielectric properties.











      ..
          !! processed by numpydoc !!


   .. py:method:: eigensolve(*args, **kwargs)

      
      Solve the eigenvalue problem of the time-modulated slab.


      :Parameters:

          **\*args** : array_like
              Arguments to be passed to `nonlinear_eigensolver`.

          **\*\*kwargs** : dict
              Keyword arguments to be passed to `nonlinear_eigensolver`.



      :Returns:

          **eigenvalues** : array_like
              The eigenvalues of the system.

          **modes** : array_like
              The eigenvectors of the system.











      ..
          !! processed by numpydoc !!


   .. py:method:: init_incident_field(omegas)

      
      Initialize the incident field.


      :Parameters:

          **omegas** : array_like
              The frequencies at which to initialize the incident field.



      :Returns:

          **incident_field** : array_like
              The initialized incident field.











      ..
          !! processed by numpydoc !!


   .. py:method:: get_incident_field(x, t, omega, Eis)

      
      Compute the incident field at the given points in space and time.


      :Parameters:

          **x** : array_like
              The points in space at which to compute the incident field.

          **t** : array_like
              The points in time at which to compute the incident field.

          **omega** : float
              The frequency at which the incident field is computed.

          **Eis** : array_like
              The Fourier coefficients of the incident field.



      :Returns:

          **Einc** : array_like
              The incident field at the specified points in space and time.











      ..
          !! processed by numpydoc !!


   .. py:method:: get_scattered_field(x, t, omega, psi, ks, modes)

      
      Compute the scattered electric field at positions x and times t.


      :Parameters:

          **x** : array_like
              The positions at which to compute the scattered field.

          **t** : array_like
              The times at which to compute the scattered field.

          **omega** : float
              The frequency of the incident wave.

          **psi** : tuple
              The coefficients of the waves inside the slab, as returned by
              extract_coefficients.

          **ks** : array_like
              The eigenvalues of the slab time-modulated medium.

          **modes** : array_like
              The eigenvectors of the slab time-modulated medium.



      :Returns:

          **E** : array_like
              The scattered electric field at positions x and times t.











      ..
          !! processed by numpydoc !!


   .. py:method:: animate_field(x, t, E, fig_ax=None)

      
      Create an animation of the electric field over time within the slab.


      :Parameters:

          **x** : array_like
              The spatial positions at which the electric field is evaluated.

          **t** : array_like
              The temporal points at which the electric field is evaluated.

          **E** : array_like
              The electric field values at the specified positions and times.

          **fig_ax** : tuple, optional
              A tuple containing a matplotlib figure and axes. If None, a new figure
              and axes are created.



      :Returns:

          **ani** : matplotlib.animation.FuncAnimation
              The animation object displaying the evolution of the electric field.











      ..
          !! processed by numpydoc !!


   .. py:method:: get_modes_normalization(modes_right, modes_left, matrix_derivative)


   .. py:method:: normalize(modes_right, modes_left, matrix_derivative)


   .. py:method:: scalar_product(modes_right, modes_left, eigenvalue_right, eigenvalue_left, matrix_right, matrix_left, matrix_derivative, diag=True)


