#pragma once
#include <iostream>

#define BOOST_ASIO_HAS_CO_AWAIT
#define BOOST_ASIO_HAS_STD_COROUTINE
#include <coroutine>
#include <boost/asio.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/awaitable.hpp>


using tcp = boost::asio::ip::tcp;
namespace test_coroutine {
    class AsyncTCPClient {
public:
    AsyncTCPClient(boost::asio::io_context& io_context)
        : resolver_(io_context), socket_(io_context) {}

    // Asynchronous connection to the server
    boost::asio::awaitable<void> connect(const std::string& host, unsigned short port) {
        auto results = co_await resolver_.async_resolve(host, std::to_string(port), boost::asio::use_awaitable);
        co_await boost::asio::async_connect(socket_, results, boost::asio::use_awaitable);
    }

    // Asynchronous write operation
    boost::asio::awaitable<void> write(const std::string& data) {
        co_await boost::asio::async_write(socket_, boost::asio::buffer(data), boost::asio::use_awaitable);
    }

    // Asynchronous read operation
    boost::asio::awaitable<std::string> read(int max_length) {
        std::string data;
        data.resize(max_length);
        
        size_t read_length = co_await socket_.async_read_some(boost::asio::buffer(data), boost::asio::use_awaitable);
        co_return data.substr(0, read_length);
    }

private:
    tcp::resolver resolver_;
    tcp::socket socket_;
};

int run() {
    boost::asio::io_context io_context;

    AsyncTCPClient client(io_context);

    boost::asio::co_spawn(io_context, [&client]() -> boost::asio::awaitable<void> {
        try {
            co_await client.connect("www.baidu.com", 80);
            co_await client.write("GET / HTTP/1.1\r\nHost: www.baidu.com\r\nConnection: close\r\n\r\n");
            std::string response = co_await client.read(1024);
            std::cout << "Received: " << response << std::endl;
        } catch (std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }, boost::asio::detached);

    io_context.run();

    return 0;
}

}



