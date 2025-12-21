use thoosa_run_config_args_derive::ThoosaRunConfig;
#[run_config_opt]
pub struct SingleField {
    #[run_config]
    pub val: i32,
}
pub struct SingleFieldRangeOrValue {
    pub val: thoosa_shared::RangeOrValue<i32>,
}
#[automatically_derived]
impl ::core::fmt::Debug for SingleFieldRangeOrValue {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field1_finish(
            f,
            "SingleFieldRangeOrValue",
            "val",
            &&self.val,
        )
    }
}
impl thoosa_run_config_args::RunConfigRangeOrValue for SingleFieldRangeOrValue {}
impl thoosa_shared::run_configs::RunConfig for SingleField {
    fn set_id(&mut self, id: thoosa_shared::run_configs::RunId) {
        self.id.replace(id);
    }
    fn type_name() -> &'static str {
        "SingleField"
    }
}
pub enum SingleFieldFields {
    Val,
}
#[automatically_derived]
impl ::core::fmt::Debug for SingleFieldFields {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::write_str(f, "Val")
    }
}
#[automatically_derived]
impl ::core::marker::StructuralPartialEq for SingleFieldFields {}
#[automatically_derived]
impl ::core::cmp::PartialEq for SingleFieldFields {
    #[inline]
    fn eq(&self, other: &SingleFieldFields) -> bool {
        true
    }
}
#[automatically_derived]
impl ::core::clone::Clone for SingleFieldFields {
    #[inline]
    fn clone(&self) -> SingleFieldFields {
        SingleFieldFields::Val
    }
}
impl std::fmt::Display for SingleFieldFields {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Val => f.write_fmt(format_args!("val")),
        }
    }
}
impl thoosa_shared::hyperparameter::FieldName for SingleFieldFields {}
impl SingleFieldRangeOrValue {
    pub fn fill_run_configs<T: thoosa_shared::hyperparameter::FieldName, F>(
        &self,
        append: F,
        const_params: &mut Vec<thoosa_shared::val::NamedVal<T>>,
        grid_params: &mut Vec<thoosa_shared::val::NamedRange<T>>,
        opt_params: &mut Vec<thoosa_shared::hyperparameter::Hyperparameter<T>>,
    )
    where
        F: Fn(SingleFieldFields) -> T,
    {
        thoosa_shared::add_range_or_value::<
            i32,
            T,
        >(
            append(SingleFieldFields::Val),
            &self.val,
            const_params,
            grid_params,
            opt_params,
        );
    }
}
impl thoosa_run_config_args::OptRunConfigSource for SingleFieldRangeOrValue {
    type P = SingleFieldFields;
    fn opt_run_configs(
        &self,
    ) -> thoosa_shared::run_configs::OptRunConfigs<
        Vec<thoosa_shared::val::NamedVal<SingleFieldFields>>,
        Vec<thoosa_shared::val::NamedRange<SingleFieldFields>>,
        Vec<thoosa_shared::hyperparameter::Hyperparameter<SingleFieldFields>>,
        SingleFieldFields,
    > {
        let mut const_params = Vec::new();
        let mut grid_params = Vec::new();
        let mut opt_params = Vec::new();
        self.fill_run_configs(
            |v| v,
            &mut const_params,
            &mut grid_params,
            &mut opt_params,
        );
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(const_params),
                grid_params,
            ),
            opt_params,
        )
    }
}
pub struct SingleFieldBuilder {
    val: Option<i32>,
}
#[automatically_derived]
impl ::core::fmt::Debug for SingleFieldBuilder {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field1_finish(
            f,
            "SingleFieldBuilder",
            "val",
            &&self.val,
        )
    }
}
#[automatically_derived]
impl ::core::default::Default for SingleFieldBuilder {
    #[inline]
    fn default() -> SingleFieldBuilder {
        SingleFieldBuilder {
            val: ::core::default::Default::default(),
        }
    }
}
impl SingleFieldBuilder {
    pub fn add_val(&mut self, path: &SingleFieldFields, val: &thoosa_shared::val::Val) {
        match path {
            SingleFieldFields::Val => {
                self.val = Some(
                    val
                        .try_into()
                        .expect(
                            &::alloc::__export::must_use({
                                ::alloc::fmt::format(
                                    format_args!(
                                        "#field_ident field value {0:?} is not valid", val,
                                    ),
                                )
                            }),
                        ),
                );
            }
        }
    }
    pub fn build(&self) -> SingleField {
        SingleField {
            val: self.val.expect("no val"),
        }
    }
}
impl<I: IntoIterator<Item = thoosa_shared::val::NamedVal<SingleFieldFields>>> From<I>
for SingleField {
    fn from(value: I) -> Self {
        let mut builder = SingleFieldBuilder::default();
        for path_val in value {
            builder.add_val(path_val.path(), path_val.val());
        }
        builder.build()
    }
}
