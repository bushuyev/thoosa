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
#[automatically_derived]
impl ::core::clone::Clone for SingleFieldRangeOrValue {
    #[inline]
    fn clone(&self) -> SingleFieldRangeOrValue {
        SingleFieldRangeOrValue {
            val: ::core::clone::Clone::clone(&self.val),
        }
    }
}
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl<'de> _serde::Deserialize<'de> for SingleFieldRangeOrValue {
        fn deserialize<__D>(
            __deserializer: __D,
        ) -> _serde::__private228::Result<Self, __D::Error>
        where
            __D: _serde::Deserializer<'de>,
        {
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            enum __Field {
                __field0,
                __ignore,
            }
            #[doc(hidden)]
            struct __FieldVisitor;
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                type Value = __Field;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private228::Formatter,
                ) -> _serde::__private228::fmt::Result {
                    _serde::__private228::Formatter::write_str(
                        __formatter,
                        "field identifier",
                    )
                }
                fn visit_u64<__E>(
                    self,
                    __value: u64,
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        0u64 => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
                fn visit_str<__E>(
                    self,
                    __value: &str,
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        "val" => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
                fn visit_bytes<__E>(
                    self,
                    __value: &[u8],
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        b"val" => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
            }
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for __Field {
                #[inline]
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private228::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    _serde::Deserializer::deserialize_identifier(
                        __deserializer,
                        __FieldVisitor,
                    )
                }
            }
            #[doc(hidden)]
            struct __Visitor<'de> {
                marker: _serde::__private228::PhantomData<SingleFieldRangeOrValue>,
                lifetime: _serde::__private228::PhantomData<&'de ()>,
            }
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = SingleFieldRangeOrValue;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private228::Formatter,
                ) -> _serde::__private228::fmt::Result {
                    _serde::__private228::Formatter::write_str(
                        __formatter,
                        "struct SingleFieldRangeOrValue",
                    )
                }
                #[inline]
                fn visit_seq<__A>(
                    self,
                    mut __seq: __A,
                ) -> _serde::__private228::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::SeqAccess<'de>,
                {
                    let __field0 = match _serde::de::SeqAccess::next_element::<
                        thoosa_shared::RangeOrValue<i32>,
                    >(&mut __seq)? {
                        _serde::__private228::Some(__value) => __value,
                        _serde::__private228::None => {
                            return _serde::__private228::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct SingleFieldRangeOrValue with 1 element",
                                ),
                            );
                        }
                    };
                    _serde::__private228::Ok(SingleFieldRangeOrValue {
                        val: __field0,
                    })
                }
                #[inline]
                fn visit_map<__A>(
                    self,
                    mut __map: __A,
                ) -> _serde::__private228::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::MapAccess<'de>,
                {
                    let mut __field0: _serde::__private228::Option<
                        thoosa_shared::RangeOrValue<i32>,
                    > = _serde::__private228::None;
                    while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                        __Field,
                    >(&mut __map)? {
                        match __key {
                            __Field::__field0 => {
                                if _serde::__private228::Option::is_some(&__field0) {
                                    return _serde::__private228::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("val"),
                                    );
                                }
                                __field0 = _serde::__private228::Some(
                                    _serde::de::MapAccess::next_value::<
                                        thoosa_shared::RangeOrValue<i32>,
                                    >(&mut __map)?,
                                );
                            }
                            _ => {
                                let _ = _serde::de::MapAccess::next_value::<
                                    _serde::de::IgnoredAny,
                                >(&mut __map)?;
                            }
                        }
                    }
                    let __field0 = match __field0 {
                        _serde::__private228::Some(__field0) => __field0,
                        _serde::__private228::None => {
                            _serde::__private228::de::missing_field("val")?
                        }
                    };
                    _serde::__private228::Ok(SingleFieldRangeOrValue {
                        val: __field0,
                    })
                }
            }
            #[doc(hidden)]
            const FIELDS: &'static [&'static str] = &["val"];
            _serde::Deserializer::deserialize_struct(
                __deserializer,
                "SingleFieldRangeOrValue",
                FIELDS,
                __Visitor {
                    marker: _serde::__private228::PhantomData::<SingleFieldRangeOrValue>,
                    lifetime: _serde::__private228::PhantomData,
                },
            )
        }
    }
};
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for SingleFieldRangeOrValue {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private228::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            let mut __serde_state = _serde::Serializer::serialize_struct(
                __serializer,
                "SingleFieldRangeOrValue",
                false as usize + 1,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "val",
                &self.val,
            )?;
            _serde::ser::SerializeStruct::end(__serde_state)
        }
    }
};
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
    ) -> Result<
        thoosa_shared::run_configs::OptRunConfigs<
            Vec<thoosa_shared::val::NamedVal<SingleFieldFields>>,
            Vec<thoosa_shared::val::NamedRange<SingleFieldFields>>,
            Vec<thoosa_shared::hyperparameter::Hyperparameter<SingleFieldFields>>,
            SingleFieldFields,
        >,
        thoosa_shared::ThoError,
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
        if opt_params.is_empty() {
            return Err(thoosa_shared::ThoError::new("no opt params in run config"));
        }
        Ok(
            thoosa_shared::run_configs::OptRunConfigs::new(
                thoosa_shared::run_configs::GridRunConfigs::new(
                    thoosa_shared::run_configs::ConstConfigs::new(const_params),
                    grid_params,
                ),
                opt_params,
            ),
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
                                        "{0} field value {1:?} is not valid", "val", val,
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
#[run_config_opt]
pub struct Outer {
    #[run_config]
    pub inner: SingleField,
}
pub struct OuterRangeOrValue {
    pub inner: SingleFieldRangeOrValue,
}
#[automatically_derived]
impl ::core::fmt::Debug for OuterRangeOrValue {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field1_finish(
            f,
            "OuterRangeOrValue",
            "inner",
            &&self.inner,
        )
    }
}
#[automatically_derived]
impl ::core::clone::Clone for OuterRangeOrValue {
    #[inline]
    fn clone(&self) -> OuterRangeOrValue {
        OuterRangeOrValue {
            inner: ::core::clone::Clone::clone(&self.inner),
        }
    }
}
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl<'de> _serde::Deserialize<'de> for OuterRangeOrValue {
        fn deserialize<__D>(
            __deserializer: __D,
        ) -> _serde::__private228::Result<Self, __D::Error>
        where
            __D: _serde::Deserializer<'de>,
        {
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            enum __Field {
                __field0,
                __ignore,
            }
            #[doc(hidden)]
            struct __FieldVisitor;
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                type Value = __Field;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private228::Formatter,
                ) -> _serde::__private228::fmt::Result {
                    _serde::__private228::Formatter::write_str(
                        __formatter,
                        "field identifier",
                    )
                }
                fn visit_u64<__E>(
                    self,
                    __value: u64,
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        0u64 => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
                fn visit_str<__E>(
                    self,
                    __value: &str,
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        "inner" => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
                fn visit_bytes<__E>(
                    self,
                    __value: &[u8],
                ) -> _serde::__private228::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        b"inner" => _serde::__private228::Ok(__Field::__field0),
                        _ => _serde::__private228::Ok(__Field::__ignore),
                    }
                }
            }
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for __Field {
                #[inline]
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private228::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    _serde::Deserializer::deserialize_identifier(
                        __deserializer,
                        __FieldVisitor,
                    )
                }
            }
            #[doc(hidden)]
            struct __Visitor<'de> {
                marker: _serde::__private228::PhantomData<OuterRangeOrValue>,
                lifetime: _serde::__private228::PhantomData<&'de ()>,
            }
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = OuterRangeOrValue;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private228::Formatter,
                ) -> _serde::__private228::fmt::Result {
                    _serde::__private228::Formatter::write_str(
                        __formatter,
                        "struct OuterRangeOrValue",
                    )
                }
                #[inline]
                fn visit_seq<__A>(
                    self,
                    mut __seq: __A,
                ) -> _serde::__private228::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::SeqAccess<'de>,
                {
                    let __field0 = match _serde::de::SeqAccess::next_element::<
                        SingleFieldRangeOrValue,
                    >(&mut __seq)? {
                        _serde::__private228::Some(__value) => __value,
                        _serde::__private228::None => {
                            return _serde::__private228::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct OuterRangeOrValue with 1 element",
                                ),
                            );
                        }
                    };
                    _serde::__private228::Ok(OuterRangeOrValue {
                        inner: __field0,
                    })
                }
                #[inline]
                fn visit_map<__A>(
                    self,
                    mut __map: __A,
                ) -> _serde::__private228::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::MapAccess<'de>,
                {
                    let mut __field0: _serde::__private228::Option<
                        SingleFieldRangeOrValue,
                    > = _serde::__private228::None;
                    while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                        __Field,
                    >(&mut __map)? {
                        match __key {
                            __Field::__field0 => {
                                if _serde::__private228::Option::is_some(&__field0) {
                                    return _serde::__private228::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("inner"),
                                    );
                                }
                                __field0 = _serde::__private228::Some(
                                    _serde::de::MapAccess::next_value::<
                                        SingleFieldRangeOrValue,
                                    >(&mut __map)?,
                                );
                            }
                            _ => {
                                let _ = _serde::de::MapAccess::next_value::<
                                    _serde::de::IgnoredAny,
                                >(&mut __map)?;
                            }
                        }
                    }
                    let __field0 = match __field0 {
                        _serde::__private228::Some(__field0) => __field0,
                        _serde::__private228::None => {
                            _serde::__private228::de::missing_field("inner")?
                        }
                    };
                    _serde::__private228::Ok(OuterRangeOrValue {
                        inner: __field0,
                    })
                }
            }
            #[doc(hidden)]
            const FIELDS: &'static [&'static str] = &["inner"];
            _serde::Deserializer::deserialize_struct(
                __deserializer,
                "OuterRangeOrValue",
                FIELDS,
                __Visitor {
                    marker: _serde::__private228::PhantomData::<OuterRangeOrValue>,
                    lifetime: _serde::__private228::PhantomData,
                },
            )
        }
    }
};
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for OuterRangeOrValue {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private228::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            let mut __serde_state = _serde::Serializer::serialize_struct(
                __serializer,
                "OuterRangeOrValue",
                false as usize + 1,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "inner",
                &self.inner,
            )?;
            _serde::ser::SerializeStruct::end(__serde_state)
        }
    }
};
impl thoosa_run_config_args::RunConfigRangeOrValue for OuterRangeOrValue {}
impl thoosa_shared::run_configs::RunConfig for Outer {
    fn set_id(&mut self, id: thoosa_shared::run_configs::RunId) {
        self.id.replace(id);
    }
    fn type_name() -> &'static str {
        "Outer"
    }
}
pub enum OuterFields {
    Inner(SingleFieldFields),
}
#[automatically_derived]
impl ::core::fmt::Debug for OuterFields {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        match self {
            OuterFields::Inner(__self_0) => {
                ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Inner", &__self_0)
            }
        }
    }
}
#[automatically_derived]
impl ::core::marker::StructuralPartialEq for OuterFields {}
#[automatically_derived]
impl ::core::cmp::PartialEq for OuterFields {
    #[inline]
    fn eq(&self, other: &OuterFields) -> bool {
        match (self, other) {
            (OuterFields::Inner(__self_0), OuterFields::Inner(__arg1_0)) => {
                __self_0 == __arg1_0
            }
        }
    }
}
#[automatically_derived]
impl ::core::clone::Clone for OuterFields {
    #[inline]
    fn clone(&self) -> OuterFields {
        match self {
            OuterFields::Inner(__self_0) => {
                OuterFields::Inner(::core::clone::Clone::clone(__self_0))
            }
        }
    }
}
impl std::fmt::Display for OuterFields {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inner(SingleFieldFields) => f.write_fmt(format_args!("inner")),
        }
    }
}
impl thoosa_shared::hyperparameter::FieldName for OuterFields {}
impl OuterRangeOrValue {
    pub fn fill_run_configs<T: thoosa_shared::hyperparameter::FieldName, F>(
        &self,
        append: F,
        const_params: &mut Vec<thoosa_shared::val::NamedVal<T>>,
        grid_params: &mut Vec<thoosa_shared::val::NamedRange<T>>,
        opt_params: &mut Vec<thoosa_shared::hyperparameter::Hyperparameter<T>>,
    )
    where
        F: Fn(OuterFields) -> T,
    {
        self.inner
            .fill_run_configs(
                |v| append(OuterFields::Inner(v)),
                const_params,
                grid_params,
                opt_params,
            );
    }
}
impl thoosa_run_config_args::OptRunConfigSource for OuterRangeOrValue {
    type P = OuterFields;
    fn opt_run_configs(
        &self,
    ) -> Result<
        thoosa_shared::run_configs::OptRunConfigs<
            Vec<thoosa_shared::val::NamedVal<OuterFields>>,
            Vec<thoosa_shared::val::NamedRange<OuterFields>>,
            Vec<thoosa_shared::hyperparameter::Hyperparameter<OuterFields>>,
            OuterFields,
        >,
        thoosa_shared::ThoError,
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
        if opt_params.is_empty() {
            return Err(thoosa_shared::ThoError::new("no opt params in run config"));
        }
        Ok(
            thoosa_shared::run_configs::OptRunConfigs::new(
                thoosa_shared::run_configs::GridRunConfigs::new(
                    thoosa_shared::run_configs::ConstConfigs::new(const_params),
                    grid_params,
                ),
                opt_params,
            ),
        )
    }
}
pub struct OuterBuilder {
    inner: SingleFieldBuilder,
}
#[automatically_derived]
impl ::core::fmt::Debug for OuterBuilder {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field1_finish(
            f,
            "OuterBuilder",
            "inner",
            &&self.inner,
        )
    }
}
#[automatically_derived]
impl ::core::default::Default for OuterBuilder {
    #[inline]
    fn default() -> OuterBuilder {
        OuterBuilder {
            inner: ::core::default::Default::default(),
        }
    }
}
impl OuterBuilder {
    pub fn add_val(&mut self, path: &OuterFields, val: &thoosa_shared::val::Val) {
        match path {
            OuterFields::Inner(sub_path) => {
                self.inner.add_val(sub_path, val);
            }
        }
    }
    pub fn build(&self) -> Outer {
        Outer { inner: self.inner.build() }
    }
}
impl<I: IntoIterator<Item = thoosa_shared::val::NamedVal<OuterFields>>> From<I>
for Outer {
    fn from(value: I) -> Self {
        let mut builder = OuterBuilder::default();
        for path_val in value {
            builder.add_val(path_val.path(), path_val.val());
        }
        builder.build()
    }
}
