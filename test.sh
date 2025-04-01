
REPO_TAG=v1.123.0
install_immich_machine_learning () {
    major=$(echo $REPO_TAG | cut -d'.' -f1)
    minor=$(echo $REPO_TAG | cut -d'.' -f2)
}

install_immich_machine_learning 


replace_usr_src () {
    echo $MINOR_VERSION
}

replace_usr_src 