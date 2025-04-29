use ordered_float::{FloatCore, NotNan};

use crate::{Parameters, Scalar};

/// Parameters to be passed unchanged to internal recursive function
pub(crate) struct InternalParameters<T: FloatCore> {
    pub(crate) max_error2: NotNan<T>,
    pub(crate) max_radius2: NotNan<T>,
    pub(crate) allow_self_match: bool,
}

impl<T: FloatCore + Scalar> From<&Parameters<T>> for InternalParameters<T> {
    fn from(value: &Parameters<T>) -> Self {
        let Parameters {
            epsilon,
            max_radius,
            allow_self_match,
            ..
        } = *value;
        let max_error = epsilon + T::from(1).unwrap();
        let max_error2 = NotNan::new(max_error * max_error).unwrap();
        let max_radius2 = NotNan::new(max_radius * max_radius).unwrap();
        InternalParameters {
            max_error2,
            max_radius2,
            allow_self_match,
        }
    }
}
