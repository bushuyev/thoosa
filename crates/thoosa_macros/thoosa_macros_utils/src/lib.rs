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
    use std::collections::HashMap;
    use log::debug;
    use thoosa_shared::val::{NamedVal, Val};

    #[test]
    fn test_named_val_cmp(){

        struct SubStruct0{
            field_a:u32
        }

        struct SubStruct1{
            field_b:u32
        }


        enum SubType {
            Sub0(SubStruct0),
            Sub1(SubStruct1)
        }

        struct SomeStruct {
            field_00:u32,
            field_01:f32,
            sub_type:SubType
        }

        //--field-00 0 --field-01 0.1 --subfield_0.field_00 1 --subfield_0.field_01 0.1  --subfield_0.subfield_0.field_00 2...

        enum Node<'a> {
            Field(Val),
            SubField(HashMap<&'a str, Node<'a>>),
        }

        let vals = [
            NamedVal::new("field_00", 0), NamedVal::new("field_01", 0.1),
            NamedVal::new("sub-type.field_a", 1),
        ];

        // let map = vals.into_iter().fold(HashMap::new(), |mut acc, x| {
        //     debug!("");
        //
        //     acc
        // });


        impl From<&[NamedVal]> for SomeStruct {
            fn from(value: &[NamedVal]) -> Self {
                todo!()
            }
        }

    }

}