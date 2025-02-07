use syn::{Type, Path};

pub fn is_primitive(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(ident) = type_path.path.get_ident() {
            match ident.to_string().as_str() {
                "i8" | "i16" | "i32" | "i64" | "i128" |
                "u8" | "u16" | "u32" | "u64" | "u128" |
                "f32" | "f64" | "bool" | "char" => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}


#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::hash::{Hash, Hasher};
    use std::path::Path;
    use log::debug;
    use thoosa_shared::val::{NamedVal, Val};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_build_from_named_vals(){

        trait Field: Debug {
            fn eq(&self, other: &dyn Field) -> bool;
            fn as_any(&self) -> &dyn Any;
            fn build(&self, val:Val){

            }
        }


        #[derive(Debug)]
        struct FieldPath<'a> {
            path:Box<dyn Field + 'a>
        }

        impl <'a> PartialEq<Self> for FieldPath<'a> {
            fn eq(&self, other: &FieldPath<'a>) -> bool {
                self.path.eq(&*other.path)
            }
        }

        impl <'a> Eq for FieldPath<'a> {
            
        }

        impl <'a> Hash for FieldPath<'a> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                
            }
        }

        impl <'a> FieldPath<'a> {
            pub fn new(path: impl Field + 'a) -> Self {
                Self { path: Box::new(path) }
            }
        }

        #[derive(Debug)]
        struct PathVal<'a> {
            path:FieldPath<'a>,
            val:Val
        }

        impl <'a> PathVal<'a> {
            pub fn new(path: FieldPath<'a>, val: Val) -> Self {
                Self { path, val }
            }
        }

        #[derive(Debug)]
        struct SubStruct0{
            field_a:u32
        }
        #[derive(Debug, Eq, PartialEq)]
        enum SubStruct0Fields {
            FieldA
        }
        struct SubStruct0Builder{
            field_a:Option<u32>
        }

        // impl Field for SubStruct0Fields {
        //     fn eq(&self, other: &dyn Field) -> bool {
        //         // PartialEq::eq(&self as Field, other)
        //         true
        //     }
        //
        //     fn as_any(&self) -> &dyn Any {
        //         self
        //     }
        // }


        #[derive(Debug, PartialEq)]
        struct SubStruct1{
            field_b:u32
        }

        #[derive(Debug, PartialEq)]
        enum SubStruct1Fields {
            FieldB
        }
        // impl Field for SubStruct1Fields {
        //     fn eq(&self, other: &dyn Field) -> bool {
        //         true
        //     }
        //
        //     fn as_any(&self) -> &dyn Any {
        //         self
        //     }
        // }

        #[derive(Debug)]
        enum SubEnum {
            Sub0(SubStruct0),
            Sub1(SubStruct1)
        }

        #[derive(Debug, PartialEq)]
        enum SubEnumFields {
            Sub0(SubStruct0Fields),
            Sub1(SubStruct1Fields)
        }

        // impl Field for SubEnumFields {
        //     fn eq(&self, other: &dyn Field) -> bool {
        //         true
        //     }
        //     fn as_any(&self) -> &dyn Any {
        //         self
        //     }
        // }

        #[derive(Debug)]
        struct SubStructB{
            field_a:u32,
            field_b:u32
        }

        #[derive(Debug, Default)]
        struct SubStructBBuilder{
            field_a:Option<u32>,
            field_b:Option<u32>
        }

        impl SubStructBBuilder {

            fn build(&mut self) -> SubStructB {

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

        #[derive(Debug, PartialEq)]
        enum SubStructBFields {
            FieldA,
            FieldB
        }

        #[derive(Debug)]
        struct SomeStruct {
            field_00:u32,
            field_01:f32,
            sub_struct:SubStructB,
            // sub_enum: SubEnum
        }

        impl SomeStruct {
            fn builder()->SomeStructBuilder {
                SomeStructBuilder::default()
            }
        }

        #[derive(Debug, PartialEq)]
        enum SomeStructFields {
            Field00,
            Field01,
            SubStruct(SubStructBFields),
            // SubEnum(SubEnumFields)
        }

        #[derive(Debug, Default)]
        struct SomeStructBuilder {
            field_00:Option<u32>,
            field_01:Option<f32>,
            sub_struct:Option<SubStructB>,
            // sub_enum: Option<SubEnum>
        }

        impl SomeStructBuilder {

            fn build(&mut self, path_vals:&[PathVal]) ->SomeStruct {

                let mut sub_struct_b_builder = SubStructB::builder();
                for path_val in path_vals {
                    // match path_val.path {
                    //
                    // }
                }

                SomeStruct {
                    field_00: self.field_00.expect("field_00"),
                    field_01: self.field_01.expect("field_01"),
                    sub_struct: sub_struct_b_builder.build(),
                    // sub_enum: (),
                }
            }
        }

        impl Field for SomeStructFields {
            fn eq(&self, other: &dyn Field) -> bool {
                if let Some(other_ssf) = other.as_any().downcast_ref::<SomeStructFields>() {
                    self == other_ssf
                } else {
                    false
                }
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn build(&self, val:Val){
                let builder = SomeStructBuilder::default();

                match self {
                    SomeStructFields::Field00 => {
                        // builder.field_00 =
                    }
                    SomeStructFields::Field01 => {

                    }
                    SomeStructFields::SubStruct(_) => {

                    }
                    // SomeStructFields::SubEnum(_) => {
                    //
                    // }
                }
            }
        }

        //--field-00 0 --field-01 0.1 --subfield_0.field_00 1 --subfield_0.field_01 0.1  --subfield_0.subfield_0.field_00 2...

        #[derive(Debug)]
        enum Node<'a> {
            Field(Val),
            SubType(HashMap<&'a str, Node<'a>>),
        }

        // let vals = [
        //     NamedVal::new("field_00", 0), NamedVal::new("field_01", 0.1),
        //     NamedVal::new("sub-type.field_a", 1),
        // ];

        let vals = [
            PathVal::new(FieldPath::new(SomeStructFields::Field00), Val::Int(1)),                                               PathVal::new(FieldPath::new(SomeStructFields::Field01), Val::Int(1)),
            PathVal::new(FieldPath::new(SomeStructFields::SubStruct(SubStructBFields::FieldA)), Val::Int(1)),                   PathVal::new(FieldPath::new(SomeStructFields::SubStruct(SubStructBFields::FieldB)), Val::Int(1)),
            // PathVal::new(FieldPath::new(SomeStructFields::SubEnum(SubEnumFields::Sub0(SubStruct0Fields::FieldA))), Val::UInt(1)),
        ];

        debug!("vals={:?}", vals);

        let ss:SomeStruct = SomeStruct::builder().build(&vals);

        // let map:HashMap<FieldPath, Val> = vals.into_iter().fold(HashMap::new(), |mut acc, x| {
        //     debug!("x={:?}", x.path);
        //
        //     acc.insert(x.path, x.val);
        //
        //
        // //
        // //     let keys = x.name().split(".");
        // //     let  map = &mut acc;
        // //     // if keys.len() == 1{
        // //     //     map.insert(keys[0], x.val());
        // //     // }
        // //     for key in keys {
        // //         debug!("{:?}", key);
        // //     }
        // //
        //     acc
        // });

        // debug!("map={:?}", map);


        impl From<&[NamedVal]> for SomeStruct {
            fn from(value: &[NamedVal]) -> Self {
                todo!()
            }
        }

    }

}