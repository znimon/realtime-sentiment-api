"""
Simple load testing script for the sentiment analysis API.
"""

import asyncio
import time
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Any

import aiohttp


@dataclass
class LoadTestResult:
    """Load test result data structure."""

    duration: float
    status_code: int
    success: bool
    error: str = None


class LoadTester:
    """Simple load tester for the API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def single_request(self, text: str) -> LoadTestResult:
        """Make a single prediction request."""
        start_time = time.time()

        try:
            async with self.session.post(
                f"{self.base_url}/predict", json={"text": text}, timeout=30
            ) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    return LoadTestResult(
                        duration=duration, status_code=response.status, success=True
                    )
                else:
                    text = await response.text()
                    return LoadTestResult(
                        duration=duration,
                        status_code=response.status,
                        success=False,
                        error=f"HTTP {response.status}: {text}",
                    )

        except Exception as e:
            duration = time.time() - start_time
            return LoadTestResult(
                duration=duration, status_code=0, success=False, error=str(e)
            )

    async def batch_request(self, texts: list[str]) -> LoadTestResult:
        """Make a batch prediction request."""
        start_time = time.time()

        try:
            async with self.session.post(
                f"{self.base_url}/batch_predict", json={"texts": texts}, timeout=60
            ) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    return LoadTestResult(
                        duration=duration, status_code=response.status, success=True
                    )
                else:
                    text = await response.text()
                    return LoadTestResult(
                        duration=duration,
                        status_code=response.status,
                        success=False,
                        error=f"HTTP {response.status}: {text}",
                    )

        except Exception as e:
            duration = time.time() - start_time
            return LoadTestResult(
                duration=duration, status_code=0, success=False, error=str(e)
            )

    async def run_concurrent_test(
        self, text: str, num_requests: int = 100, concurrency: int = 10
    ) -> list[LoadTestResult]:
        """Run concurrent requests test."""
        print(f"Running {num_requests} requests with concurrency {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request():
            async with semaphore:
                return await self.single_request(text)

        tasks = [limited_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        return results

    async def run_batch_test(
        self, texts: list[str], num_requests: int = 50, concurrency: int = 5
    ) -> list[LoadTestResult]:
        """Run batch requests test."""
        print(f"Running {num_requests} batch requests with concurrency {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)

        async def limited_batch_request():
            async with semaphore:
                return await self.batch_request(texts)

        tasks = [limited_batch_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        return results

    def analyze_results(self, results: list[LoadTestResult]) -> dict[str, Any]:
        """Analyze test results."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        if not successful_results:
            return {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0.0,
                "error": "No successful requests",
            }

        durations = [r.duration for r in successful_results]

        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "response_times": {
                "mean": mean(durations),
                "median": median(durations),
                "min": min(durations),
                "max": max(durations),
                "std_dev": stdev(durations) if len(durations) > 1 else 0,
            },
            "throughput": len(successful_results) / sum(durations) if durations else 0,
        }

        if failed_results:
            error_counts = {}
            for result in failed_results:
                error = result.error or f"HTTP {result.status_code}"
                error_counts[error] = error_counts.get(error, 0) + 1
            analysis["errors"] = error_counts

        return analysis

    def print_analysis(self, analysis: dict[str, Any], test_type: str = "Test"):
        """Print analysis results."""
        print(f"\n=== {test_type} Results ===")
        print(f"Total requests: {analysis['total_requests']}")
        print(f"Successful requests: {analysis['successful_requests']}")
        print(f"Failed requests: {analysis['failed_requests']}")
        print(f"Success rate: {analysis['success_rate']:.2f}%")

        if "response_times" in analysis:
            rt = analysis["response_times"]
            print("\nResponse Times:")
            print(f"  Mean: {rt['mean']:.3f}s")
            print(f"  Median: {rt['median']:.3f}s")
            print(f"  Min: {rt['min']:.3f}s")
            print(f"  Max: {rt['max']:.3f}s")
            print(f"  Std Dev: {rt['std_dev']:.3f}s")
            print(f"  Throughput: {analysis['throughput']:.2f} req/s")

        if "errors" in analysis:
            print("\nErrors:")
            for error, count in analysis["errors"].items():
                print(f"  {error}: {count}")


async def main():
    """Main load test function."""
    # Test data
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Highly recommended!",
        "Worst experience ever. Don't buy this.",
        "Pretty good, would buy again.",
        "Not bad, but could be better.",
        "Outstanding quality and service!",
        "Disappointing and overpriced.",
        "Exactly what I expected.",
    ]

    batch_texts = test_texts[:5]  # Smaller batch for testing

    async with LoadTester() as tester:
        # Test 1: Single request load test
        print("Testing single requests...")
        single_results = await tester.run_concurrent_test(
            text=test_texts[0], num_requests=100, concurrency=10
        )

        single_analysis = tester.analyze_results(single_results)
        tester.print_analysis(single_analysis, "Single Request Load Test")

        # Test 2: Batch request load test
        print("\nTesting batch requests...")
        batch_results = await tester.run_batch_test(
            texts=batch_texts, num_requests=50, concurrency=5
        )

        batch_analysis = tester.analyze_results(batch_results)
        tester.print_analysis(batch_analysis, "Batch Request Load Test")

        # Test 3: Mixed load test
        print("\nTesting mixed load...")
        mixed_tasks = []

        # Add single requests
        for i in range(50):
            text = test_texts[i % len(test_texts)]
            mixed_tasks.append(tester.single_request(text))

        # Add batch requests
        for i in range(25):
            mixed_tasks.append(tester.batch_request(batch_texts))

        mixed_results = await asyncio.gather(*mixed_tasks)
        mixed_analysis = tester.analyze_results(mixed_results)
        tester.print_analysis(mixed_analysis, "Mixed Load Test")


if __name__ == "__main__":
    asyncio.run(main())
