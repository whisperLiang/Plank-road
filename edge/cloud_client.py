import threading
import time
from typing import Dict, Any
from loguru import logger

class AsyncCloudClient:
    """
    Edge-side asynchronous price cache module.
    Maintains a non-blocking communication with the cloud for latest resource shadow prices.
    """
    def __init__(self, server_ip: str, update_interval: float = 5.0):
        self.server_ip = server_ip
        self.update_interval = update_interval
        self.cached_prices: Dict[str, float] = {
            "price_comp": float('inf'),
            "price_bw": float('inf')
        }
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        """Start the background async mechanism to pull prices."""
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
        
    def stop(self) -> None:
        """Stop the background thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            
    def get_cached_prices(self) -> Dict[str, float]:
        """
        Direct memory read of cached prices. No networking operations.
        Ensures the inference main process is never blocked.
        """
        with self.lock:
            return self.cached_prices.copy()
            
    def _fetch_loop(self) -> None:
        """Periodic background task to pull prices from the cloud."""
        while self.running:
            try:
                # Abstract representation of cloud fetch.
                # E.g. using gRPC request to GlobalResourceManager
                prices = self._fetch_from_cloud()
                with self.lock:
                    self.cached_prices = prices
            except Exception as e:
                logger.warning(f"[AsyncCloudClient] Loss connection with cloud: {e}. Setting conservative state.")
                # Fault tolerance: set to infinity to go to conservative state
                with self.lock:
                    self.cached_prices = {
                        "price_comp": float('inf'),
                        "price_bw": float('inf')
                    }
            time.sleep(self.update_interval)

    def _fetch_from_cloud(self) -> Dict[str, float]:
        """
        Network call to request newest shadow prices.
        Implement actual gRPC / REST logic here.
        """
        import grpc
        try:
            from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
            channel = grpc.insecure_channel(self.server_ip)
            stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
            # Example gRPC call (modify according to actual proto file definition later)
            # req = message_transmission_pb2.Empty()
            # reply = stub.get_shadow_prices(req, timeout=3.0)
            # return {"price_comp": reply.price_comp, "price_bw": reply.price_bw}
        except ImportError:
            pass
        
        # Simulated successful return
        return {"price_comp": 5.0, "price_bw": 2.0}
