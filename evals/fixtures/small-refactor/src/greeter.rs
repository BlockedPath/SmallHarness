pub fn greet_user(name: &str) {
    println!("Hello, {name}!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets() {
        greet_user("test");
    }
}
