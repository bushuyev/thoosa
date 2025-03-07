use syn::{Type, Path, Field, Lit, PathArguments, TypePath, GenericArgument, Ident};
use syn::__private::ToTokens;

pub fn is_primitive(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(ident) = type_path.path.get_ident() {
            return  _is_primitive(ident);
        }
    }
    false
}

fn _is_primitive(ident: &Ident) -> bool {
    match ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "i64" | "i128" |
        "u8" | "u16" | "u32" | "u64" | "u128" |
        "f32" | "f64" | "bool" | "char" => {
             true
        }
        _ => {
            false
        }
    }
}

pub fn is_option_primitive(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(first_segment) = path.segments.first() {
            if first_segment.ident == "Option" {
                if let PathArguments::AngleBracketed(ref args) = first_segment.arguments {
                    if let Some(GenericArgument::Type(Type::Path(inner_type_path))) = args.args.first() {
                        if let Some(inner_ident) = inner_type_path.path.get_ident() {
                            return _is_primitive(inner_ident);
                        }
                    }
                }
            }
        }
    }
    false
}

pub fn get_option_inner_ty(ty: &Type) -> Option<Type> {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(first_segment) = path.segments.first() {
            if first_segment.ident == "Option" {
                if let PathArguments::AngleBracketed(ref args) = first_segment.arguments {
                    if let Some(GenericArgument::Type(inner_type)) = args.args.first() {
                       return Some(inner_type.clone())
                    }
                }
            }
        }
    }
    None
}


pub fn comment(field: &&Field, field_name: &String) -> String {
    let mut descr = field_name.clone();
    for attr in &field.attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(lit_str) = &meta.value {
                    // let doc_comment = lit_str.lit.to_string();
                    if let Lit::Str(token) = &lit_str.lit {
                        descr = token.value();
                    }
                }
            }
        }
    }
    descr
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::HashMap;
    use std::fmt::{Debug, Display, Formatter};
    use std::hash::{Hash, Hasher};
    use std::path::Path;
    use std::str::FromStr;
    use log::debug;
    use thoosa_shared::hyperparameter::{Hyperparameter, FieldName};
    use thoosa_shared::{add_prefix, add_range_or_value, RangeOrValue, ThoosaError};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_build_from_named_vals() {


        #[derive(Debug, Clone)]
        struct SubStruct0 {
            field_a: u32,
        }

        #[derive(Debug, Clone, Eq, PartialEq)]
        enum SubStruct0Fields {
            FieldA
        }
        impl Display for SubStruct0Fields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "SubStruct0Fields::FieldA")
            }
        }

        #[derive(Debug, Default)]
        struct SubStruct0Builder {
            field_a: Option<u32>,
        }

        impl SubStruct0Builder {

            fn build(&self)-> SubStruct0 {
                SubStruct0 {
                    field_a: self.field_a.expect("SubStruct0Fields::FieldA is required"),
                }
            }

            fn add_val(&mut self, path: &SubStruct0Fields, val: &Val)  {
                match path {
                    SubStruct0Fields::FieldA => {
                        self.field_a = Some(val.try_into().expect("SubStruct0Fields::FieldA is required"));
                    }
                }
            }
        }

        #[derive(Debug, PartialEq, Clone)]
        struct SubStruct1 {
            field_b: u32,
        }

        #[derive(Debug, Clone, PartialEq)]
        enum SubStruct1Fields {
            FieldB
        }

        impl Display for SubStruct1Fields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "SubStruct1Fields::FieldB")
            }
        }

        #[derive(Debug, Default)]
        struct SubStruct1Builder {
            field_b:Option<u32>,
        }

        impl SubStruct1Builder {
            fn build(&self)-> SubStruct1 {
                SubStruct1 {
                    field_b: self.field_b.expect("SubStruct1Fields::FieldB is required"),
                }
            }

            fn add_val(&mut self, path: &SubStruct1Fields, val: &Val)  {
                match path {
                    SubStruct1Fields::FieldB => {
                        self.field_b = Some(val.try_into().expect("SubStruct1Fields::FieldB is required"));
                    }
                }
            }
        }

        #[derive(Debug, Clone)]
        enum SubEnum {
            Sub0(SubStruct0),
            Sub1(SubStruct1),
        }

        #[derive(Debug, Clone, PartialEq)]
        enum SubEnumFields {
            Sub0(SubStruct0Fields),
            Sub1(SubStruct1Fields),
        }

        impl Display for SubEnumFields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    SubEnumFields::Sub0(v) => {
                        write!(f, "SubEnum::Sub0/{v}")
                    }
                    SubEnumFields::Sub1(v) => {
                        write!(f, "SubEnum::Sub1/{v}")
                    }
                }
            }
        }

        #[derive(Debug, Default)]
        struct SubEnumBuilder {
            sub_struct0_builder:SubStruct0Builder,
            sub_struct1_builder:SubStruct1Builder,
            sub_enum:Option<SubEnumBuilderEnum>

        }
        #[derive(Debug, PartialEq)]
        enum SubEnumBuilderEnum {
            Sub0,
            Sub1
        }

        impl SubEnumBuilder {

            fn build(&self)-> SubEnum{

                match self.sub_enum.as_ref().expect("SubEnumBuilder::sub_enum is required") {
                    SubEnumBuilderEnum::Sub0 => {
                        SubEnum::Sub0(self.sub_struct0_builder.build())
                    }
                    SubEnumBuilderEnum::Sub1 => {
                        SubEnum::Sub1(self.sub_struct1_builder.build())
                    }
                }
            }

            fn add_val(&mut self, path: &SubEnumFields, val: &Val)  {
                match path {
                    SubEnumFields::Sub0(path) => {
                        let prev = self.sub_enum.replace(SubEnumBuilderEnum::Sub0);
                        assert!(prev.as_ref().map(|p|p==&SubEnumBuilderEnum::Sub0).unwrap_or(true), "mixed fields from SubEnum: prev={prev:?} and Sub0, path={path:?} ");

                        self.sub_struct0_builder.add_val(path, val);
                    }
                    SubEnumFields::Sub1(path) => {
                        let prev = self.sub_enum.replace(SubEnumBuilderEnum::Sub1);
                        assert!(prev.as_ref().map(|p|p==&SubEnumBuilderEnum::Sub1).unwrap_or(true), "mixed fields from SubEnum: prev={prev:?} and Sub1, path={path:?} ");

                        self.sub_struct1_builder.add_val(path, val);
                    }
                }
            }
        }


        #[derive(Debug)]
        struct SubStructB {
            field_a: u32,
            field_b: u32,
        }

        #[derive(Debug, Default)]
        struct SubStructBBuilder {
            field_a: Option<u32>,
            field_b: Option<u32>,
        }

        impl SubStructBBuilder {
            fn add_val(&mut self, path: &SubStructBFields, val: &Val) /*->SomeStruct*/ {
                match path {
                    SubStructBFields::FieldA => {
                        self.field_a = Some(val.try_into().expect("field A not valid"));
                    }
                    SubStructBFields::FieldB => {
                        self.field_b = Some(val.try_into().expect("field B not valid"));
                    }
                }
            }

            fn build(&self) -> SubStructB {
                SubStructB {
                    field_a: self.field_a.expect("field_a"),
                    field_b: self.field_b.expect("field_b"),
                }
            }
        }

        impl SubStructB {
            fn builder() -> SubStructBBuilder {
                SubStructBBuilder::default()
            }
        }

        #[derive(Debug, Clone, PartialEq)]
        enum SubStructBFields {
            FieldA,
            FieldB,
        }

        impl Display for SubStructBFields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    SubStructBFields::FieldA => {
                        write!(f, "SubStructBFields::FieldA")
                    }
                    SubStructBFields::FieldB => {
                        write!(f, "SubStructBFields::FieldA")
                    }
                }
            }
        }

        impl FieldName for SubStructBFields {}

        #[derive(Debug)]
        struct SubStructBRangeOrValue {
            field_a: RangeOrValue<u32>,
            field_b: RangeOrValue<u32>,
        }

        impl SubStructBRangeOrValue {
            pub(crate) fn fill_run_configs<P: FieldName>(&self, prefix: Option<&P>, const_params: &mut Vec<NamedVal<P>>, grid_params: &mut Vec<NamedRange<P>>, opt_params: &mut Vec<Hyperparameter<P>>) {
                // add_range_or_value::<u32, SubStructBFields>(add_prefix(prefix, SubStructBFields::FieldA), &self.field_a, const_params, grid_params, opt_params);
                todo!()
            }
        }

        #[derive(Debug)]
        struct SomeStruct {
            field_00: u32,
            field_01: f32,
            sub_struct: SubStructB,
            sub_enum: SubEnum
        }

        #[derive(Debug)]
        struct SomeStructRangeOrValue {
            field_00: RangeOrValue<u32>,
            field_01: RangeOrValue<f32>,
            sub_struct: SubStructBRangeOrValue,
            // sub_enum: SubEnum
        }

        impl SomeStructRangeOrValue {

            pub(crate) fn fill_run_configs<P: FieldName>(&self, const_params: &mut Vec<NamedVal<P>>, grid_params: &mut Vec<NamedRange<P>>, opt_params: &mut Vec<Hyperparameter<P>>) {
                //TODO
            }
        }

        impl SomeStruct {
            fn builder() -> SomeStructBuilder {
                SomeStructBuilder::default()
            }
        }

        #[derive(Debug, Clone, PartialEq)]
        enum SomeStructFields {
            Field00,
            Field01,
            SubStruct(SubStructBFields),
            SubEnum(SubEnumFields)
        }

        impl FieldName for SomeStructFields {}

        impl Display for SomeStructFields {

            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    SomeStructFields::Field00 => {
                        write!(f, "SomeStructFields::Field00")
                    }
                    SomeStructFields::Field01 => {
                        write!(f, "SomeStructFields::Field01")
                    }
                    SomeStructFields::SubStruct(v) => {
                        write!(f, "SomeStructFields::SubStruct/{v}")
                    }
                    SomeStructFields::SubEnum(v) => {
                        write!(f, "SomeStructFields::SubEnum/{v}")
                    }
                }
            }
        }


        #[derive(Debug, Default)]
        struct SomeStructBuilder {
            field_00: Option<u32>,
            field_01: Option<f32>,
            sub_struct_builder: SubStructBBuilder,
            sub_enum_builder: SubEnumBuilder
        }

        impl SomeStructBuilder {

            fn add_path_vals(path_vals: impl IntoIterator<Item=NamedVal<SomeStructFields>>) -> SomeStruct {
                let mut builder = SomeStructBuilder::default();

                for path_val in path_vals {
                    builder.add_val(path_val.path(), path_val.val());
                }

                builder.build()
            }

            fn add_val(&mut self, path: &SomeStructFields, val: &Val) {
                match path {
                    SomeStructFields::Field00 => {
                        self.field_00 = Some(val.try_into().expect("field 00 not valid"));
                    }
                    SomeStructFields::Field01 => {
                        self.field_01 = Some(val.try_into().expect("field 01 not valid"));
                    }
                    SomeStructFields::SubStruct(sub_path) => {
                        self.sub_struct_builder.add_val(sub_path, val);
                    }
                    SomeStructFields::SubEnum(sub_path) => {
                        self.sub_enum_builder.add_val(sub_path, val)
                    }
                }
            }

            fn build(&self) -> SomeStruct {
                SomeStruct {
                    field_00: self.field_00.expect("field_00"),
                    field_01: self.field_01.expect("field_01"),
                    sub_struct: self.sub_struct_builder.build(),
                    sub_enum: self.sub_enum_builder.build(),
                }
            }
        }

        let some_struct_range_or_value = SomeStructRangeOrValue {
            field_00: RangeOrValue::Value(1),
            field_01: RangeOrValue::Value(0.1),
            sub_struct: SubStructBRangeOrValue {
                field_a: RangeOrValue::Value(2),
                field_b: RangeOrValue::Value(3),
            },
        };


        let mut const_params:Vec<NamedVal<SomeStructFields>> = Vec::new();
        let mut grid_params:Vec<NamedRange<SomeStructFields>> = Vec::new();
        let mut opt_params:Vec<Hyperparameter<SomeStructFields>> = Vec::new();

        some_struct_range_or_value.fill_run_configs(&mut const_params, &mut grid_params, &mut opt_params);

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Val::Int(1)), NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldA), Val::Int(3)), NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldB), Val::Int(4)),
            NamedVal::new(SomeStructFields::SubEnum(SubEnumFields::Sub0(SubStruct0Fields::FieldA)), Val::UInt(1)),
        ];

        let some_struct: SomeStruct = SomeStructBuilder::add_path_vals(path_vals);

        assert_eq!(some_struct.field_00, 1);
        assert_eq!(some_struct.field_01, 2.0);
        assert_eq!(some_struct.sub_struct.field_a, 3);
        assert_eq!(some_struct.sub_struct.field_b, 4);
        let SubEnum::Sub0(sub_struct_0) = some_struct.sub_enum else {
            panic!("some_struct.sub_enum should be Sub0");
        };
        assert_eq!(sub_struct_0.field_a, 1);

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Val::Int(1)), NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldA), Val::Int(3)), NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldB), Val::Int(4)),
            NamedVal::new(SomeStructFields::SubEnum(SubEnumFields::Sub1(SubStruct1Fields::FieldB)), Val::UInt(2)),
        ];

        let some_struct: SomeStruct = SomeStructBuilder::add_path_vals(path_vals);
        assert_eq!(some_struct.field_00, 1);
        assert_eq!(some_struct.field_01, 2.0);
        assert_eq!(some_struct.sub_struct.field_a, 3);
        assert_eq!(some_struct.sub_struct.field_b, 4);
        let SubEnum::Sub1(sub_struct_1) = some_struct.sub_enum else {
            panic!("some_struct.sub_enum should be Sub0");
        };
        assert_eq!(sub_struct_1.field_b, 2);


    }

}