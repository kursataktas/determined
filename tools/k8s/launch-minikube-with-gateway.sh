#!/bin/bash

set -e

K8S_VERSION=${K8S_VERSION:-1.29.5} # https://endoflife.date/kubernetes
minikube_profile=$1
this_dir=$(dirname "$(realpath "$0")")
contour_config_file=$this_dir/gateways/contour-provisioner.yaml

if [ -z "$1" ]; then
    echo "Usage: $0 <minikube_profile>"
    exit 1
fi

if [ ! -f $contour_config_file ]; then
    echo "Gateway configuration file not found: $contour_config_file"
    exit 1
fi

minikube start --profile $minikube_profile --kubernetes-version $K8S_VERSION
kubectl apply -f $contour_config_file

kubectl apply -f - <<EOF
kind: GatewayClass
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: contour
spec:
  controllerName: projectcontour.io/gateway-controller
---
kind: Gateway
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: contour
  namespace: projectcontour
spec:
  gatewayClassName: contour
  listeners:
    - name: tcp
      protocol: TCP
      port: 52335 # Need at least one listener on a gateway. Master will add and patch to it.
      allowedRoutes:
        namespaces:
          from: All
EOF

if sudo -n true 2>/dev/null; then
    # Either like have a smaller subnet so we don't conflict. Or like don't start it for the second one.
    nohup minikube --profile $minikube_profile tunnel & # TODO won't work for users with sudo passwords.
else
    echo "sudo password is required to start the tunnel."
    echo "Please run the following command separately to start the tunnel:"
    echo "minikube --profile $minikube_profile tunnel"
    read -p "Press [Enter] once the tunnel has started..."
fi

for ((i = 0; i < 60; i++)); do
    export GATEWAY_IP=$(kubectl -n projectcontour get svc envoy-contour -o=jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$GATEWAY_IP" ]; then
        echo "External IP address of envoy-contour service: $GATEWAY_IP"
        break
    fi

    sleep 1
done
